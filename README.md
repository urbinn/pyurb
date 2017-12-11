# pyurb

In de volgende folder vind je delen van het project:
- pyurb: python libraries van Urb
- notebooks: notebooks met exeprimenten
- bundleadjustment: C++ code om g2o te gebruiken met PYBIND11 bindings

Om lokaal te draaien moet je:
./compile

Dan wordt de C++ code in bundleadjustment compiled en een urbg2o.so gemaakt in de pyurb directory.

Om bij de pyurb libraries te gebruiken vanuit de notebooks moet de root van dit project aan het system-path worden toegevoegd (zie match-sequence2.ipynb), dat lossen we later nog wat mooier op.
