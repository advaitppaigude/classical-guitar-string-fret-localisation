SIM = verilator
TOPLEVEL_LANG = verilog

TB_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
RTL_DIR := $(abspath $(TB_DIR)/../../rtl)

COCOTB_TOPLEVEL = full_classifier
COCOTB_TEST_MODULES = full_classifier_functional_validation
VERILOG_SOURCES = $(RTL_DIR)/model_pkg.sv \
                  $(RTL_DIR)/full_classifier.sv \
                  $(RTL_DIR)/load_model.sv \
                  $(RTL_DIR)/parametrised_note_classifier_fsm.sv \
                  $(RTL_DIR)/pipelined_parametrised_multiply_accumulate.sv \
                  $(RTL_DIR)/compare.sv

BITS ?= 6
COMPILE_ARGS += -GBITS=$(BITS)
# The runner invokes make from the format's simulation directory.
SIM_BUILD ?= build
COCOTB_RESULTS_FILE ?= $(SIM_BUILD)/results.xml
export PYTHONPATH := $(TB_DIR):$(PYTHONPATH)

include $(shell cocotb-config --makefiles)/Makefile.sim
