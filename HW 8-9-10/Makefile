PYTHON ?= python3
.PHONY: check contracts integration check-hw67
check: contracts integration
contracts:
	"$(PYTHON)" -m unittest discover -s tests -v
integration:
	"$(PYTHON)" -m unittest discover -s integration_tests -v
check-hw67:
	cd '../HW 6-7' && "$(PYTHON)" -m unittest discover -s tests -v
