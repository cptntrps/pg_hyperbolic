EXTENSION = pg_hyperbolic
MODULE_big = pg_hyperbolic
OBJS = src/hyperbolic.o
DATA = sql/pg_hyperbolic--0.1.0.sql
REGRESS = distance index_hnsw stability

PG_CFLAGS += -O2 -ftree-vectorize -fassociative-math \
             -fno-signed-zeros -fno-trapping-math

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
