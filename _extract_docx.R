extract_docx <- function(path) {
  tmp <- tempfile()
  unzip(path, files = "word/document.xml", exdir = tmp)
  xml_file <- file.path(tmp, "word", "document.xml")
  lines <- readLines(xml_file, warn = FALSE)
  txt <- paste(lines, collapse = " ")
  txt <- gsub("<w:p[ >]", "\n", txt)
  txt <- gsub("<[^>]+>", "", txt)
  txt <- gsub("[ \t]+", " ", txt)
  txt <- gsub("\n ", "\n", txt)
  txt <- gsub("\n{3,}", "\n\n", txt)
  trimws(txt)
}

cat("=== b1_DePeer.docx ===\n")
cat(extract_docx("b1_DePeer.docx"))
cat("\n\n=== b2_DePeer.docx ===\n")
cat(extract_docx("b2_DePeer.docx"))
cat("\n")
