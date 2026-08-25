def derive_gene_name(annotation, used_gene_names, unique_id_counter):
    """Derive a unique gene name from annotation, matching Panaroo's logic.

    Shared by generate_output (gene_presence_absence) and
    alignment_functions.generate_alignments (alignment file names) so that the
    two agree on what every node is called. Callers must iterate nodes in the
    same order -- both use `ORDER BY node_id` -- or the group_N counters drift.
    """
    if annotation:
        name = "~~~".join(
            gn for gn in annotation.strip().strip(";").split(";") if gn != ""
        )
        name = "".join(e for e in name if e.isalnum() or e in ["_", "~"])
    else:
        name = ""

    if name and name.lower() not in used_gene_names:
        used_gene_names.add(name.lower())
        return name, unique_id_counter

    gen_name = f"group_{unique_id_counter}"
    unique_id_counter += 1
    used_gene_names.add(gen_name.lower())
    return gen_name, unique_id_counter
