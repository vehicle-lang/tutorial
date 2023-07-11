SOURCES+=templates/header.md
SOURCES+=0.\ Introduction.md
SOURCES+=1.\ Vehicle\ Language.md
SOURCES+=2.\ Proving\ Robustness.md
SOURCES+=3.\ Property-Driven\ Training.md
SOURCES+=templates/footer.md
SOURCES+=bibliography.bib
SOURCES+=$(wildcard images/*.png)

################################################################################
# Markdown
################################################################################

README.md: $(SOURCES)
	pandoc                              \
		--defaults table-of-contents.yaml \
		--to html                         \
		--output index.html

################################################################################
# HTML
################################################################################

.PHONY: html
html: index.html

index.html: $(SOURCES)
	pandoc                                                  \
		--defaults table-of-contents.yaml                     \
		--to html                                             \
		--metadata title="A Vehicle Tutorial"                 \
		--standalone                                          \
		--template templates/easy_template/easy_template.html \
		--table-of-contents                                   \
		--katex                                               \
		--output index.html

.PHONY: view
view: index.html
	python -m http.server
