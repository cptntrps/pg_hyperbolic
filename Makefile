EXTENSION = pg_hyperbolic
MODULE_big = pg_hyperbolic
OBJS = src/hyperbolic.o src/geometric.o src/aggregate.o src/curvature.o
DATA = sql/pg_hyperbolic--0.1.0.sql sql/pg_hyperbolic--0.1.1.sql \
       sql/pg_hyperbolic--0.1.0--0.1.1.sql sql/pg_hyperbolic--0.1.1--0.1.0.sql
REGRESS = midpoint ivfflat
REGRESS_OPTS = --inputdir=test

PG_CFLAGS += -O2 -ftree-vectorize -fassociative-math \
             -fno-signed-zeros -fno-trapping-math

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
