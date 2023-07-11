SOURCES+=templates/header.md
SOURCES+=chapters/1.\ Introduction.md
SOURCES+=chapters/2.\ Vehicle\ Language.md
SOURCES+=chapters/3.\ Proving\ Robustness.md
SOURCES+=chapters/4.\ Property-Driven\ Training.md
SOURCES+=templates/footer.md
SOURCES+=bibliography.bib
SOURCES+=$(wildcard images/*.png)

################################################################################
# HTML
################################################################################

.PHONY: html
html: _site/index.html

_site/:
	mkdir -p _site/

_site/index.html: $(SOURCES) | _site/
	cp -R images _site/images
	pandoc                                                  \
		--defaults table-of-contents.yaml                     \
		--to html                                             \
		--metadata title="A Vehicle Tutorial"                 \
		--standalone                                          \
		--template templates/easy_template/easy_template.html \
		--table-of-contents                                   \
		--number-sections                                     \
		--katex                                               \
		--output _site/index.html

.PHONY: view
view: _site/index.html
	python -m http.server --directory _site/


################################################################################
# Markdown
################################################################################

.PHONY: gfm
gfm: README.md

README.md: $(SOURCES)
	pandoc                              \
		--defaults table-of-contents.yaml \
		--to gfm                          \
		--output README.md
