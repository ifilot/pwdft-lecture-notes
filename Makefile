LATEXMK ?= latexmk
LATEXMK_FLAGS ?= -pdf -interaction=nonstopmode -halt-on-error -file-line-error

MAIN := main
JOBNAME := pwdft-filot
PDF := $(JOBNAME).pdf
TEX_SOURCES := $(MAIN).tex \
	$(wildcard config/*.tex) \
	$(wildcard sections/*.tex) \
	$(wildcard img/*.tex)
FIGURES := $(wildcard img/*.pdf)
BIB_SOURCES := $(wildcard *.bib)

.DEFAULT_GOAL := all

.PHONY: all pdf watch clean distclean help

all: pdf

pdf: $(PDF)

$(PDF): $(TEX_SOURCES) $(FIGURES) $(BIB_SOURCES)
	$(LATEXMK) $(LATEXMK_FLAGS) -jobname=$(JOBNAME) $(MAIN).tex

watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -jobname=$(JOBNAME) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -c -jobname=$(JOBNAME) $(MAIN).tex

distclean:
	$(LATEXMK) -C -jobname=$(JOBNAME) $(MAIN).tex

help:
	@echo "Available targets:"
	@echo "  all/pdf    Build $(PDF) (default)"
	@echo "  watch      Rebuild whenever a source file changes"
	@echo "  clean      Remove auxiliary LaTeX files"
	@echo "  distclean  Remove auxiliary files and $(PDF)"
	@echo "  help       Show this help"
