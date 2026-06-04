# KTOOLS example: CH2O + OH -> pre-reactive complex (microcanonical VTST entrance).
# `make` writes the ktools.dat (input-gen smoke test; no MultiWell binary needed).
# `make solver` additionally runs the ktools binary on the generated deck.
all: run

run: PES.in database.json
	gausskit run $< --verbose --dry

solver: run
	cd ktools_PES_1 && ktools ktools.dat

clean:
	rm -rf ktools_PES_*
