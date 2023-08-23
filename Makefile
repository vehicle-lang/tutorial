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
		--lua-filter filters/pygmentize.lua                   \
		--output _site/index.html

.PHONY: view
view: _site/index.html
	python -m http.server --directory _site/

################################################################################
# PDF
################################################################################

.PHONY: pdf
pdf: tutorial.pdf

tutorial.pdf: tutorial.tex
	latexmk -pdflua -latexoption=-shell-escape tutorial.tex

.PHONY: latex
latex: tutorial.tex

tutorial.tex: $(SOURCES)
	pandoc                                     \
		--defaults table-of-contents.yaml        \
		--to latex                               \
		--metadata title="A Vehicle Tutorial"    \
		--standalone                             \
		--table-of-contents                      \
		--number-sections                        \
		--include-in-header templates/minted.tex \
		--lua-filter filters/minted.lua          \
		--output tutorial.tex

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

################################################################################
# Clean
################################################################################

.PHONY: clean
clean:
	rm -rf _site/
	rm -rf _minted-tutorial/
	rm -rf svg-inkscape/
	latexmk -C tutorial
	rm tutorial.bbl
	rm tutorial.tex
