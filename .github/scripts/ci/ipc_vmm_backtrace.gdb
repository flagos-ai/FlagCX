set pagination off
set confirm off
set print thread-events off
handle SIGSEGV stop print nopass
run
if $_isvoid($_exitcode)
  thread apply all bt full
  info registers
  x/16i $pc-24
  info sharedlibrary
  info proc mappings
end
