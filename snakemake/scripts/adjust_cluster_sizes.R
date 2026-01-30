#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

# arg parser
get_arg <- function(flag) {
  idx <- match(flag, args)
  if (is.na(idx)) stop(paste("Missing argument:", flag))
  args[idx + 1]
}

paths_file       <- get_arg("--assemblies")
poppunk_dir    <- get_arg("--poppunk-dir")
min_count        <- as.integer(get_arg("--min"))
max_count        <- as.integer(get_arg("--max"))
outdir <- get_arg("--outdir")

# get paths to assemblies from poppunk input file (paths are in second column, sampleids are in first column)
paths <- read.delim(paths_file, header=F) 

# get all poppunk clusters (from bashscript output)
assemblies <- read.csv(file = paste0(outdir, "/combined_clusters.csv"), header = F) 
assemblies <- assemblies[order(assemblies$V2), ]

# add paths to sampleids in poppunk clusters csv
index <- which(is.na(match(paths$V1, assemblies$V1))==F)
paths <- paths[index,]
m <- match(assemblies$V1, paths$V1)
assemblies$paths <- paths$V2[m] # changed this sort function, possibly not correct anymore, double check

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
split_clusters <- names(counts)[split]

for (cl in split_clusters) {
  idx <- which(assemblies$V2 == cl)
  n <- length(idx)
  half <- ceiling(n / 2)

  # split roughly in half 
  assemblies$V2[idx[1:half]] <- paste0(cl, "a")
  assemblies$V2[idx[(half + 1):n]] <- paste0(cl, "b")
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
