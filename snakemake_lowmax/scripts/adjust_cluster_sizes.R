#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

# arg parser
get_arg <- function(flag) {
  idx <- match(flag, args)
  if (is.na(idx)) stop(paste("Missing argument:", flag))
  args[idx + 1]
}

paths_file       <- get_arg("--assemblies")
assemblies <- get_arg("--combined-clusters")
poppunk_dir    <- get_arg("--poppunk-dir")
min_count        <- as.integer(get_arg("--min"))
max_count        <- as.integer(get_arg("--max"))
outdir <- get_arg("--outdir")

# get paths to assemblies from poppunk input file (paths are in second column, sampleids are in first column)
paths <- read.delim(paths_file, header=F) 

# get all poppunk clusters (from bashscript output)
assemblies <- read.csv(file = assemblies, header = F) 
assemblies <- assemblies[order(assemblies$V2), ]

# add paths to sampleids in poppunk clusters csv
index <- which(is.na(match(paths$V1, assemblies$V1))==F)
paths <- paths[index,]
m <- match(assemblies$V1, paths$V1)
assemblies$paths <- paths$V2[m] # changed this sort function, possibly not correct anymore, double check

# remove NAs and missing strings
assemblies <- assemblies[!is.na(assemblies$paths) & assemblies$paths != "", ]

# get counts of each cluster
counts <- table(assemblies$V2)

# min and max samples per cluster
min <- min_count
max <- max_count

merge <- which(counts < min) # to be merged together
split <- which(counts > max) # to be spit in two

# check that merged file won't be too large
merged_size <- 0
for (m in merge) {
  merged_size <- merged_size + counts[m]
}

# create merged clusters
merge_clusters <- names(counts)[merge]
merged_name <- "merged"
merged_isolates <- paste(merge_clusters, collapse = "_")
assemblies$V2[assemblies$V2 %in% merge_clusters] <- merged_name

# create split clusters
# NB: recompute counts AFTER merging, so the "merged" bucket is itself split when it
# exceeds max_count. The original script took split_clusters from the pre-merge `counts`,
# which meant "merged" could never be split however large it grew -- that is what left
# s_pneumoniae with a 2096-genome "merged" cluster that never finished ggcaller.
counts_after_merge <- table(assemblies$V2)
split_clusters <- names(counts_after_merge)[which(counts_after_merge > max_count)]

# sub-cluster suffix pool: two-letter aa,ab,...,az,ba,...,zz (676 total).
# Uniform two-letter suffixes so split clusters sort cleanly (e.g. 1aa, 1ab, ...).
# Supports up to 676 sub-clusters per cluster, enough for the dominant clonal
# cluster in M. tuberculosis (~47 pieces at max=1500).
suffix_pool <- paste0(rep(letters, each = 26), letters)

for (cl in split_clusters) {
  idx <- which(assemblies$V2 == cl)
  n <- length(idx)
  n_pieces <- ceiling(n / max_count)
  if (n_pieces > length(suffix_pool)) {
    stop(sprintf("Cluster %s needs %d sub-clusters; suffix scheme supports max %d. Lower max_count or extend suffixes.",
                 cl, n_pieces, length(suffix_pool)))
  }
  chunk_size <- ceiling(n / n_pieces)
  pieces <- ((seq_len(n) - 1) %/% chunk_size) + 1
  assemblies$V2[idx] <- paste0(cl, suffix_pool[pieces])
}

# check that merged file won't be too large

# get counts of each cluster
counts <- table(assemblies$V2)
merged_size <- sum(counts["merged"])

# double check the result
table(assemblies$V2)

# write ggcaller inputs
for (cluster in unique(assemblies$V2)) {
  write.table(x = assemblies$paths[assemblies$V2==cluster],
            file = paste0(outdir, "/sizebalanced_cluster_", cluster, ".txt"),
            col.names = F, row.names = F, quote = F)
}

# write merged index
write.table(sort(unique(assemblies$V2)), file = paste0(outdir, "/sizebalanced_clusters_index.csv"),
            col.names = F, row.names = F, quote = F)
