.PHONY: all model-submission ppt clean

all: model-submission ppt

model-submission: model_submission.pdf

ppt: presentation.pdf

model_submission.pdf: model_submission.tex
	@pdflatex model_submission.tex
	@echo "Model Architecture Documentation Compiled: model_submission.pdf"

presentation.pdf: presentation.tex
	@pdflatex presentation.tex
	@echo "Presentation Compiled: presentaion.pdf"

clean: 
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.synctex.gz || true
