
SIM = verilator
TOPLEVEL_LANG = verilog

TB_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# One elaboration per format: make BITS=5, make BITS=6, or make BITS=8.
BITS ?= 6
FORMAT_5 := b5_f2
FORMAT_6 := b6_f3
FORMAT_8 := b8_f4
FORMAT := $(FORMAT_$(BITS))
ifeq ($(FORMAT),)
$(error Unsupported BITS=$(BITS); choose 5, 6, or 8)
endif

TOPLEVEL = $(FORMAT)_fsm_v1_pipelined
MODULE = single_classifier_functional_validation

VERILOG_SOURCES = $(abspath $(TB_DIR)/../../rtl/$(TOPLEVEL).sv) \
                  $(abspath $(TB_DIR)/../../rtl/pipelined_parametrised_multiply_accumulate.sv) \
                  $(abspath $(TB_DIR)/../../rtl/compare.sv)

COMPILE_ARGS += -GBITS=$(BITS)
SIM_BUILD ?= sim_build/classifier_b$(BITS)
COCOTB_RESULTS_FILE ?= $(SIM_BUILD)/results.xml
export PYTHONPATH := $(TB_DIR):$(PYTHONPATH)

include $(shell cocotb-config --makefiles)/Makefile.sim
