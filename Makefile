.PHONY: install hw1-generate hw1-bench hw1-check hw1-test hw2-inspect hw2-check hw2-test check
.NOTPARALLEL:

install:
	uv sync --locked

hw1-generate:
	$(MAKE) -C "HW 1" generate

hw1-bench:
	$(MAKE) -C "HW 1" bench

hw1-check:
	$(MAKE) -C "HW 1" check

hw1-test:
	$(MAKE) -C "HW 1" test

hw2-inspect:
	$(MAKE) -C "HW 2" inspect

hw2-check:
	$(MAKE) -C "HW 2" check

hw2-test:
	$(MAKE) -C "HW 2" test

check:
	$(MAKE) hw1-check
	$(MAKE) hw2-check
