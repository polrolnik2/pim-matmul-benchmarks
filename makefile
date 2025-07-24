# Docker image name
DOCKER_IMAGE := pim-matmul-dev

# Build the Docker image
docker-build:
	docker build -q --platform linux/amd64 -t $(DOCKER_IMAGE) .

CC ?= gcc

# Use project root from environment variable
ROOT ?= $(PIM_MATMUL_BENCHMARKS_ROOT)

# Helper to extract sources from YAML
DEPS_YAML := $(ROOT)/defn/dependencies.yaml
DEPS_SRCS := $(shell python3 -c "import yaml,sys; print(' '.join(yaml.safe_load(open('$(DEPS_YAML)'))['sources']))" 2>/dev/null || echo "")
INCLUDE_DIRS := $(shell python3 -c "import yaml; print(' '.join('-I'+d for d in yaml.safe_load(open('$(DEPS_YAML)'))['include_dirs']))" 2>/dev/null || echo "")

CFLAGS += $(INCLUDE_DIRS)

SimplePIM:
	git clone --depth 1 --filter=blob:none --sparse https://github.com/CMU-SAFARI/SimplePIM.git lib/simplepim
	cd lib/simplepim && git sparse-checkout set lib
	mv lib/simplepim/lib/* lib/simplepim/
	rm -rf lib/simplepim/lib

clean:
	rm -rf lib/simplepim
	rm -rf $(BIN_DIR)

BIN_DIR := bin

# Helper to extract unittest sources from YAML
UNITTEST_SRCS := $(shell python3 -c "import yaml; print(' '.join(yaml.safe_load(open('defn/unittests.yaml'))['unittest']))" 2>/dev/null || echo "")
UNITTEST_BINS := $(addprefix bin/,$(basename $(notdir $(UNITTEST_SRCS))))

# Ensure bin/ exists
bin:
	@mkdir -p bin

build-unittests: $(UNITTEST_BINS)

run-unittests: docker-build
	@mkdir -p scratch; \
	set -e; \
	for t in $(UNITTEST_SRCS); do \
	  echo "Building and running $$t"; \
	  docker run --rm --platform linux/amd64 -v $(CURDIR):/workspace $(DOCKER_IMAGE) bash -c \
		". /opt/upmem-2025.1.0-Linux-x86_64/upmem_env.sh simulator && \
		. /workspace/source.me && \
		make -C tests run FILE=$$(basename $$t)"; \
	done

.PHONY: SimplePIM clean build-unittests run-unittests