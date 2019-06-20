.PHONY: env-reqs

env:
	virtualenv env
	env/bin/pip install -r requirements.txt

env-reqs: env
	env/bin/pip install -r requirements.txt

clean-env:
	rm -rf env
