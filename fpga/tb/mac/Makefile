
SIM = verilator
TOPLEVEL_LANG = verilog

TB_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# One elaboration per format.
BITS ?= 6
FORMAT_5 := b5_f2
FORMAT_6 := b6_f3
FORMAT_8 := b8_f4
FORMAT_10 := b10_f5
FORMAT := $(FORMAT_$(BITS))
ifeq ($(FORMAT),)
$(error Unsupported BITS=$(BITS); choose 5, 6, 8, or 10)
endif

ifeq ($(BITS),10)
TOPLEVEL = parametrised_note_classifier_fsm
else
TOPLEVEL = $(FORMAT)_fsm_v1_pipelined
endif
MODULE = single_classifier_functional_validation

VERILOG_SOURCES = $(abspath $(TB_DIR)/../../rtl/$(TOPLEVEL).sv) \
                  $(abspath $(TB_DIR)/../../rtl/pipelined_parametrised_multiply_accumulate.sv) \
                  $(abspath $(TB_DIR)/../../rtl/compare.sv)

COMPILE_ARGS += -GBITS=$(BITS)
SIM_BUILD ?= sim_build/classifier_b$(BITS)
COCOTB_RESULTS_FILE ?= $(SIM_BUILD)/results.xml
export PYTHONPATH := $(TB_DIR):$(PYTHONPATH)

include $(shell cocotb-config --makefiles)/Makefile.sim
