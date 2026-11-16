# Shared project-header dependencies for Device IR. The including Makefile
# sets OBJDIR and IR_INPUT_BC before including this file.
.DEFAULT_GOAL := all
IR_DEPFILE := $(OBJDIR)/libflagcx_device.d
IR_DEPFLAGS := -MMD -MP -MF $(IR_DEPFILE) -MT $(IR_INPUT_BC)

-include $(IR_DEPFILE)

$(IR_INPUT_BC): Makefile ../device_ir_dependencies.mk
