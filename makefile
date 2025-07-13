SimplePIM:
	git clone --depth 1 --filter=blob:none --sparse https://github.com/CMU-SAFARI/SimplePIM.git lib/simplepim
	cd lib/simplepim && git sparse-checkout set lib
	mv lib/simplepim/lib/* lib/simplepim/
	rm -rf lib/simplepim/lib

clean:
	rm -rf lib/simplepim

.PHONY: SimplePIM clean