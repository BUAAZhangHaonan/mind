GLSim code isolation verified on 2026-04-13:
- run_glsim.py exits early, produces no output files.
- glsim.py exports no active implementation.
- glsim_adapted.py is not imported by any maintained script.
- export_paper_package.py does not package GLSim outputs.
GLSim is permanently excluded from this project because its experimental paradigm conflicts with the object yes/no benchmark design used by MIND. No GLSim experiments will be performed.
