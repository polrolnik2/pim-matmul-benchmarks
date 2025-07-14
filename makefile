CC ?= gcc

# Helper to extract sources from YAML
DEPS_YAML := defn/dependencies.yaml
DEPS_SRCS := $(shell python3 -c "import yaml,sys; print(' '.join(yaml.safe_load(open('$(DEPS_YAML)'))['sources']))" 2>/dev/null || echo "")

# Helper to extract include dirs from YAML and convert to -I flags
INCLUDE_DIRS := $(shell python3 -c "import yaml; print(' '.join('-I'+d for d in yaml.safe_load(open('defn/dependencies.yaml'))['include_dirs']))" 2>/dev/null || echo "")

CFLAGS += $(INCLUDE_DIRS)

SimplePIM:
	git clone --depth 1 --filter=blob:none --sparse https://github.com/CMU-SAFARI/SimplePIM.git lib/simplepim
	cd lib/simplepim && git sparse-checkout set lib
	mv lib/simplepim/lib/* lib/simplepim/
	rm -rf lib/simplepim/lib

clean:
	rm -rf lib/simplepim

BIN_DIR := bin

# Compile and run all C unittests in tests/
UNITTEST_SRCS := $(wildcard tests/*-unittests.c)
UNITTEST_BINS := $(patsubst tests/%.c,$(BIN_DIR)/%,$(UNITTEST_SRCS))

build-unittests: $(UNITTEST_BINS)

run-unittests: build-unittests
	@set -e; for t in $(UNITTEST_BINS); do echo "Running $$t"; ./$$t; done

$(BIN_DIR)/%: tests/%.c $(DEPS_SRCS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< $(DEPS_SRCS) -o $@

$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

.PHONY: SimplePIM clean build-unittests run-unittests