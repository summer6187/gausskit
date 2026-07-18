# CH2O + OH barrierless capture: harmonic vs VPT2 vs Vhrd2 (rigid-rotor) VTST.
#
# One deck (ktools_harm11.dat, 11 trial dividing surfaces 2.5->5.5 A), three DOS treatments of
# the soft transitional modes, all through the SAME gausskit-native mu-VTST engine (`gausskit vtst`,
# pure Python -- no MultiWell binary needed). Shows VPT2 COLLAPSING the barrierless entrance rate
# and the scan-matched rigid-rotor (Vhrd2) RECOVERING it. See README.md.
#
#   make rates   run the three curves -> rates/engine_{harmonic,vpt2,modeaware}.csv
#   make plot    rates + plot_kt.py   -> kt.png
#   make clean   remove rates/ + kt.png

PY   ?= python
DECK := ktools_harm11.dat

.PHONY: all rates plot clean help
.DEFAULT_GOAL := help

help:                 ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-9s\033[0m %s\n", $$1, $$2}'

rates: rates/engine_harmonic.csv rates/engine_vpt2.csv rates/engine_modeaware.csv  ## all three k(T)

rates/engine_harmonic.csv: $(DECK)          ## 11 surfaces, all modes harmonic (native-KTOOLS equivalent)
	@mkdir -p rates
	gausskit vtst $(DECK) -o $@

rates/engine_vpt2.csv: $(DECK) dens/rp_25_vpt2.dens  ## VPT2 where computable (only rp_25 has a valid X-matrix)
	@mkdir -p rates
	gausskit vtst $(DECK) --dens TS2=dens/rp_25_vpt2.dens --zpe-shift TS2=-183.25 -o $@

rates/engine_modeaware.csv: $(DECK) vtst_modeaware.args  ## soft modes as scan-matched rigid rotors (Vhrd2)
	@mkdir -p rates
	gausskit vtst $(DECK) $$(cat vtst_modeaware.args) -o $@

plot: rates            ## the three-curve comparison figure
	$(PY) plot_kt.py

all: plot              ## rates + figure

clean:                 ## remove regenerable outputs
	rm -rf rates kt.png
