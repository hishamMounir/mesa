# SPDX-License-Identifier: MIT
#
# G17 (Apple M5) machine-code layer: tables generated from g17_isa.py.

g17_gen = custom_target(
  'g17_isa',
  input : ['g17_gen.py'],
  output : ['g17_isa.h', 'g17_isa.c', 'g17_golden.h'],
  command : [
    prog_python, '@INPUT@', '@OUTDIR@',
    '--golden', join_paths(meson.current_source_dir(), 'test', 'g17_golden_vectors.json'),
  ],
  depend_files : files('g17_isa.py', 'g17_length.json', 'test/g17_golden_vectors.json'),
)

idep_g17_isa_h = declare_dependency(
  sources : [g17_gen[0]],
  include_directories : include_directories('.'),
)

libasahi_g17 = static_library(
  'asahi_g17',
  ['g17_pack.c', g17_gen[1]],
  dependencies : [idep_g17_isa_h],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

if with_tests
  test(
    'g17_tests',
    executable(
      'g17_tests',
      ['test/g17_test.c', g17_gen[2]],
      dependencies : [idep_g17_isa_h],
      link_with : [libasahi_g17],
    ),
    suite : ['asahi'],
  )
endif
