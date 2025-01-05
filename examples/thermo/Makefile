all: output run

output: database.json

database.json:
	gausskit output ../data -o $@

run: PES.in
	gausskit run $< --verbose

clean:
	rm -r thermo_PES_*
