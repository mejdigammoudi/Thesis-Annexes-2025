# ---- WRDS Cloud RStudio: Build dynamic S&P500 (2014-2024) from WRDS, no CSV ----
# Reproducible Data Engineering for Part II §4 (no shrcd anywhere)
suppressPackageStartupMessages({
  library(DBI)
  library(RPostgres)
  library(dplyr)
  library(readr)
  library(stringr)
  library(lubridate)
  library(glue)
  library(data.table)
  library(purrr)
})

# --------- Parameters ---------
START  <- as.Date("2014-01-01")
END    <- as.Date("2024-12-31")
OUTDIR <- "data_engineering_outputs"
dir.create(OUTDIR, showWarnings = FALSE)

# --------- 1) Connect to WRDS ---------
wrds <- dbConnect(RPostgres::Postgres(),
                  host = 'wrds-pgdata.wharton.upenn.edu',
                  port = 9737,
                  dbname = 'wrds',
                  user = Sys.getenv("mejdigammoudi"),
                  password = 'amWSAZq20130!',
                  sslmode = 'require')

on.exit(try(dbDisconnect(wrds), silent = TRUE), add = TRUE)

# --------- 2) Helper: check if a table exists ---------
wrds_table_exists <- function(schema, table) {
  DBI::dbExistsTable(wrds, DBI::Id(schema = schema, table = table))
}

# --------- 3) Locate a CRSP S&P500 membership table ---------
candidates <- list(
  c("crsp","msp500list"),
  c("crsp","sp500list"),
  c("crsp","dspsp500list"),
  c("crsp","mspsp500list")
)

candidate_tbl <- NULL
for (ct in candidates) {
  if (wrds_table_exists(ct[[1]], ct[[2]])) {
    candidate_tbl <- ct
    break
  }
}
if (is.null(candidate_tbl)) {
  stop("❌ No CRSP S&P 500 membership table found (e.g., crsp.msp500list). Check WRDS permissions.")
}
schema <- candidate_tbl[[1]]
table  <- candidate_tbl[[2]]
message("✅ Membership table found: ", schema, ".", table)

# --------- 4) Pull membership and standardize entry/exit columns ---------
mem_df <- dbGetQuery(wrds, glue("select * from {schema}.{table}"))
mem_df <- as.data.frame(mem_df)
cn <- tolower(names(mem_df))
names(mem_df) <- cn

if (!("permno" %in% cn)) stop("❌ Membership table has no PERMNO column.")

start_candidates <- c("start", "from", "namedt", "startdt", "in", "in_date", "effdt")
end_candidates   <- c("ending","thru","nameendt","enddt","out","out_date","expdt")

pick_col <- function(cands, cn) {
  found <- cands[cands %in% cn]
  if (length(found) == 0) NA_character_ else found[[1]]
}
start_col <- pick_col(start_candidates, cn)
end_col   <- pick_col(end_candidates, cn)

if (is.na(start_col) | is.na(end_col)) {
  stop("❌ Could not identify membership start/end columns in ", schema, ".", table, ". Inspect names(mem_df).")
}

mem_df <- mem_df %>%
  transmute(
    permno = as.integer(permno),
    entry_date_raw = as.Date(.data[[start_col]]),
    exit_date_raw  = as.Date(.data[[end_col]])
  ) %>%
  filter(!is.na(permno), !is.na(entry_date_raw))

# Restrict membership to study window and close open-ended exits
mem_df <- mem_df %>%
  mutate(
    entry_date = pmax(entry_date_raw, START),
    exit_date  = coalesce(exit_date_raw, as.Date("2099-12-31")),
    exit_date  = pmin(exit_date, END)
  ) %>%
  filter(entry_date <= exit_date) %>%
  select(permno, entry_date, exit_date)

# --------- 5) CRSP stocknames (alias nameenddt -> nameendt) ---------
stocknames <- tryCatch({
  dbGetQuery(wrds, "
    select permno, permco, comnam, namedt, nameenddt as nameendt, ticker, ncusip, siccd
    from crsp.stocknames
  ")
}, error = function(e) {
  dbGetQuery(wrds, "
    select permno, permco, comnam, namedt, nameendt, ticker, ncusip, siccd
    from crsp.stocknames
  ")
})

stocknames <- stocknames %>%
  mutate(
    ticker   = toupper(str_trim(ticker)),
    namedt   = as.Date(namedt),
    nameendt = as.Date(nameendt)
  )

# --------- 6) Overlap membership windows with name-validity windows ---------
setDT(mem_df)
setDT(stocknames)
setkey(stocknames, permno, namedt, nameendt)
setkey(mem_df,     permno, entry_date, exit_date)

cons_map <- foverlaps(
  x = mem_df[, .(permno, entry_date, exit_date,
                 entry_date2 = entry_date, exit_date2 = exit_date)],
  y = stocknames[, .(permno, namedt, nameendt, ticker, comnam, ncusip, siccd)],
  by.x = c("permno","entry_date","exit_date"),
  by.y = c("permno","namedt","nameendt"),
  type = "any", nomatch = 0L
)

cons_map <- cons_map[, .(
  permno, ticker, comnam, ncusip, siccd,
  entry_date = pmax(entry_date, namedt),
  exit_date  = pmin(exit_date,  nameendt)
)]

cons_map[, overlap_len := as.integer(exit_date - entry_date)]
setorder(cons_map, permno, -overlap_len)
cons_map <- unique(cons_map, by = c("permno","entry_date","exit_date"))

# Save constituents/mapping
cons <- cons_map[, .(ticker, entry_date, exit_date)]
cons <- cons[!is.na(ticker)]
cons <- cons[entry_date <= exit_date]
cons <- cons[order(ticker, entry_date)]

write_csv(as.data.frame(cons), file.path(OUTDIR, "A1_sp500_constituents_with_entry_exit.csv"))
write_csv(as.data.frame(cons[, .(ticker)] %>% unique() %>% arrange(ticker)),
          file.path(OUTDIR, "sp500_tickers_unique.csv"))
fwrite(cons_map[order(ticker, entry_date)],
       file.path(OUTDIR, "sp500_tickers_with_permnos.csv"))

# --------- 7) Pull CRSP DSF (daily), study window (no shrcd) ---------
sql_dsf <- glue("
  select permno, date, openprc, prc as close, vol as volume, ret as crsp_ret
  from crsp.dsf
  where date between '{START}' and '{END}'
")
dsf <- dbGetQuery(wrds, sql_dsf) %>%
  mutate(date = as.Date(date),
         close = abs(as.numeric(close)))

# Keep only PERMNOs from our S&P 500 membership mapping
valid_permno <- unique(cons_map$permno)
dsf <- dsf %>% filter(permno %in% valid_permno)

write_csv(dsf, file.path(OUTDIR, "sp500_only_prices_2014_2024.csv"))

# --------- 8) Attach names by date-window and in-index flag ---------
dsf_names <- dsf %>%
  inner_join(
    as.data.frame(stocknames[, .(permno, comnam, namedt, nameendt, ticker, ncusip, siccd)]),
    by = "permno"
  ) %>%
  filter(namedt <= date, nameendt >= date)

dsf_names <- dsf_names %>%
  inner_join(as.data.frame(unique(cons[, .(ticker, entry_date, exit_date)])),
             by = "ticker") %>%
  mutate(in_index_flag = (date >= entry_date & date <= exit_date))

# Resolve duplicates: keep most recent name record per (permno, date)
setDT(dsf_names)
setorder(dsf_names, permno, date, namedt)
dsf_names <- unique(dsf_names, by = c("permno","date"), fromLast = TRUE)

panel_full <- dsf_names %>%
  select(
    permno, date, openprc, close, volume, crsp_ret,
    ticker, comnam, ncusip, siccd,
    entry_date, exit_date, in_index_flag
  ) %>%
  arrange(permno, date)

write_csv(panel_full, file.path(OUTDIR, "sp500_prices_with_names_2014_2024_FULL.csv"))

# Optional: per-year splits
panel_full %>%
  mutate(year = year(date)) %>%
  group_split(year) %>%
  walk(function(df) {
    y <- unique(df$year)
    write_csv(df %>% select(-year), file.path(OUTDIR, glue("sp500_prices_with_names_{y}.csv")))
  })

# --------- 9) Annex A helper summaries ---------
yr_summary <- panel_full %>%
  distinct(ticker, year = year(date)) %>%
  count(year, name = "n_distinct_tickers")
write_csv(yr_summary, file.path(OUTDIR, "A3_yearly_distinct_tickers.csv"))

turnover <- bind_rows(
  as.data.frame(cons[, .(year = year(entry_date), event = "entry")]),
  as.data.frame(cons[, .(year = year(exit_date),  event = "exit")])
) %>%
  drop_na(year) %>%
  count(year, event) %>%
  tidyr::pivot_wider(names_from = event, values_from = n, values_fill = 0)
write_csv(turnover, file.path(OUTDIR, "A4_turnover_entries_exits_by_year.csv"))

message("✅ Completed: outputs saved in ", OUTDIR)
