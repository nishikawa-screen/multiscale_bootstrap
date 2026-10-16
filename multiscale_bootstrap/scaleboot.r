## Rscript wrapper for scaleboot::sbfit.
##
## Usage:
##   Rscript scaleboot.r --temp_dir=<dir> --models=<comma-separated>
##     --k=<int> --sk=<int> --select=<average|best>
##
## Reads from <temp_dir>:
##   bp.csv     - one row per hypothesis, no header, first column =
##                hypothesis_id, remaining columns = bootstrap probability
##                at each valid scale (column order must match scales.csv
##                row order).
##   scales.csv - header "nb,sa", one row per valid scale, in the same
##                column order as bp.csv.
##
## Writes to <temp_dir>:
##   result.csv - columns hypothesis_id, au_pvalue, au_se, si_pvalue,
##                si_se, best_model, beta0, beta1, one row per
##                hypothesis.
##   best_model is the minimum-AIC model from summary(), regardless of
##   --select. P-values, standard errors and beta use --select instead.

if (!suppressWarnings(require(scaleboot, quietly = TRUE))) {
    stop("The R package 'scaleboot' is required but not installed.")
}

temp_dir <- NULL
models <- NULL
k <- NULL
sk <- NULL
select <- NULL

args <- commandArgs(trailingOnly = TRUE)
for (arg in args) {
    s <- strsplit(arg, "=")[[1]]
    if (length(s) < 2) {
        next
    }
    key <- s[1]
    value <- paste(s[2:length(s)], collapse = "=")

    if (key == "--temp_dir") {
        temp_dir <- value
    } else if (key == "--models") {
        models <- strsplit(value, ",")[[1]]
    } else if (key == "--k") {
        k <- as.integer(value)
    } else if (key == "--sk") {
        sk <- as.integer(value)
    } else if (key == "--select") {
        select <- value
    }
}

if (is.null(temp_dir) || is.null(models) || is.null(k) || is.null(sk) || is.null(select)) {
    stop("Usage: Rscript scaleboot.r --temp_dir=<dir> --models=<comma-separated> --k=<int> --sk=<int> --select=<average|best>")
}

bp_path <- file.path(temp_dir, "bp.csv")
meta_path <- file.path(temp_dir, "scales.csv")
out_path <- file.path(temp_dir, "result.csv")

bp_df <- read.csv(bp_path, header = FALSE, row.names = 1, check.names = FALSE)
hypothesis_ids <- rownames(bp_df)
bp_mat <- as.matrix(bp_df)
storage.mode(bp_mat) <- "double"

meta <- read.csv(meta_path)
nb <- meta$nb
sa <- meta$sa

pv_row <- paste0("k.", k)
spv_row <- paste0("sk.", sk)

fit <- sbfit(bp_mat, nb = nb, sa = sa, models = models, names.hp = hypothesis_ids)
smry <- summary(fit, k = k, sk = sk, hypothesis = "alternative")
pv <- sbpval(smry, select = select)

best_model <- vapply(hypothesis_ids, function(hid) {
    m <- smry[[hid]]$best$model
    if (is.null(m)) NA_character_ else as.character(m)
}, character(1))

# Indexed explicitly by hypothesis_ids: pv's column order is not
# guaranteed to match, so this fixes the row order of result.csv.
result <- data.frame(
    hypothesis_id = hypothesis_ids,
    au_pvalue = as.numeric(pv$pvalue[pv_row, hypothesis_ids]),
    au_se = as.numeric(pv$pvalue.sd[pv_row, hypothesis_ids]),
    si_pvalue = as.numeric(pv$pvalue[spv_row, hypothesis_ids]),
    si_se = as.numeric(pv$pvalue.sd[spv_row, hypothesis_ids]),
    best_model = best_model,
    beta0 = as.numeric(pv$beta["beta0", hypothesis_ids]),
    beta1 = as.numeric(pv$beta["beta1", hypothesis_ids])
)

write.csv(result, out_path, row.names = FALSE)
