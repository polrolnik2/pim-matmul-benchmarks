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

# Extract all runtime parameters as compiler flags
PARAMS_YAML := $(ROOT)/defn/params.yaml
RUNTIME_PARAM_FLAGS := $(shell python3 -c "import yaml; params=yaml.safe_load(open('$(PARAMS_YAML)')); print(' '.join([f'-D{k}={v}' for item in params.get('runtime_params', []) for k, v in item.items()]))" 2>/dev/null || echo "")

CFLAGS += $(INCLUDE_DIRS)
CFLAGS += $(RUNTIME_PARAM_FLAGS)

SimplePIM:
	@if [ ! -d lib/simplepim ]; then \
		git clone --depth 1 --filter=blob:none --sparse https://github.com/CMU-SAFARI/SimplePIM.git lib/simplepim && \
		cd lib/simplepim && git sparse-checkout set lib && \
		mv lib/simplepim/lib/* lib/simplepim/ && \
		rm -rf lib/simplepim/lib; \
	fi

clean:
	rm -rf lib/simplepim
	rm -rf $(BIN_DIR)
	rm -rf $(PIM_MATMUL_BENCHMARKS_ROOT)/scratch/

BIN_DIR := bin

# Helper to extract unittest sources from YAML
UNITTEST_SRCS := $(shell python3 -c "import yaml; print(' '.join(yaml.safe_load(open('defn/unittests.yaml'))['unittest']))" 2>/dev/null || echo "")
UNITTEST_BINS := $(addprefix bin/,$(basename $(notdir $(UNITTEST_SRCS))))

# Ensure bin/ exists
bin:
	@mkdir -p bin

build-unittests: $(UNITTEST_BINS)

run-unittests: docker-build SimplePIM bin build-dpu
	@mkdir -p scratch; \
	set -e; \
	for t in $(UNITTEST_SRCS); do \
	  echo "Building and running $$t"; \
	  docker run --rm --platform linux/amd64 -v $(CURDIR):/workspace $(DOCKER_IMAGE) bash -c \
		". /opt/upmem-2025.1.0-Linux-x86_64/upmem_env.sh simulator && \
		. /workspace/source.me && \
		make -C tests run FILE=$$(basename $$t)"; \
	done; \
	python3 scripts/parse_unittest_logs.py

# Build DPU binaries
build-dpu: docker-build bin
	docker run --rm --platform linux/amd64 -v $(CURDIR):/workspace $(DOCKER_IMAGE) bash -c \
		". /opt/upmem-2025.1.0-Linux-x86_64/upmem_env.sh simulator && \
		. /workspace/source.me && \
		dpu-upmem-dpurte-clang -O2 $(RUNTIME_PARAM_FLAGS) -g -o /workspace/bin/matrix_multiply_dpu /workspace/src/dpu/pim_dpu_matrix_multiply.c -I src"

.PHONY: SimplePIM clean build-unittests run-unittests build-dpu