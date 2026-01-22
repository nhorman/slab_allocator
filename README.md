# Experimental OpenSSL slab allocator

This is just an experiment to create a DSO that hooks into the OpenSSL malloc/realloc/free routines to allow for slab allocation of small objects to prevent haivng to stress the OS allocator too much
