# KTOOLS example: CH2O + OH -> pre-reactive complex (microcanonical VTST entrance).
# `make`        writes the ktools.dat (input-gen smoke test; no MultiWell binary needed).
# `make solver` additionally runs the ktools binary on the generated deck.
# `make plot`   runs the solver, then compares our k(T) against MultiWell's bundled
#               ch2o+oh reference (references/ch2o+oh/) -> compare_kt.png.
all: run

run: PES.in database.json
	gausskit run $< --verbose --dry

solver: run
	cd ktools_PES_1 && ktools ktools.dat

plot: solver
	python plot_compare.py

clean:
	rm -rf ktools_PES_* compare_kt.png
