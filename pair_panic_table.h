/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(panic/table,PairPanicTable)

#else

#ifndef LMP_PAIR_PANIC_TABLE_H
#define LMP_PAIR_PANIC_TABLE_H

#include "pair.h"

namespace LAMMPS_NS {

class PairPanicTable : public Pair {
 public:
  PairPanicTable(class LAMMPS *);
  virtual ~PairPanicTable();

  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

 protected:
  enum{LOOKUP,LINEAR,SPLINE,BITMAP};

  int tabstyle,tablength;
  struct Table {
    int wall_ninput,opening_ninput;                 // Guillermo Frank - 15 june 2018
    int ninput,rflag,fpflag,match,ntablebits;
    int nshiftbits,nmask;
    double rlo,rhi,fplo,fphi,cut;
    double *rfile,*efile,*ffile;
    double *e2file,*f2file;
    double innersq,delta,invdelta,deltasq6;
    double *rsq,*drsq,*e,*de,*f,*df,*e2,*f2,*rsqt,*drsqt; // Sticco 18 Jun 2018
    double *w1,*w2,*w3,*w4,*w5;                     // Guillermo Frank - 15 june 2018
    double *c1,*c2,*c3,*c4,*c5,*c6,*c7,*c8,*c9;     // Guillermo Frank - 15 june 2018
    double *op1,*op2,*op3,*op4,*op5,*op6;           // Guillermo Frank - 15 june 2018
  };
  int ntables;
  Table *tables;

  int arg_wall = 14;        // Sticco 18 Jun 2018
  double vd;                // Sticco 18 Jun 2018

  int **tabindex;

  virtual void allocate();
  void read_table(Table *, char *, char *);
  void param_extract(Table *, char *);
  void bcast_table(Table *);
  void spline_table(Table *);
  virtual void compute_table(Table *);
  void null_table(Table *);
  void free_table(Table *);
  void spline(double *, double *, int, double, double, double *);
  double splint(double *, double *, double *, int, double);

  void wall_extract(Table *, char *,FILE *);        // Guillermo Frank - 15 june 2018
  void compute_wallforce(double, double, double, double, double force_wall[], Table *);  // Sticco 18 Jun 2018
  void compute_desiredforce(double x, double y, double vx, double vy, double* force_desired, double vd, Table *tb); // Sticco 18 Jun 2018

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Pair distance < table inner cutoff

Two atoms are closer together than the pairwise table allows.

E: Pair distance > table outer cutoff

Two atoms are further apart than the pairwise table allows.

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Unknown table style in pair_style command

Style of table is invalid for use with pair_style table command.

E: Illegal number of pair table entries

There must be at least 2 table entries.

E: Invalid pair table length

Length of read-in pair table is invalid

E: Invalid pair table cutoff

Cutoffs in pair_coeff command are not valid with read-in pair table.

E: Bitmapped table in file does not match requested table

Setting for bitmapped table in pair_coeff command must match table
in file exactly.

E: All pair coeffs are not set

All pair coefficients must be set in the data file or by the
pair_coeff command before running a simulation.

E: Cannot open file %s

The specified file cannot be opened.  Check that the path and name are
correct. If the file is a compressed file, also check that the gzip
executable can be found and run.

E: Did not find keyword in table file

Keyword used in pair_coeff command was not found in table file.

E: Bitmapped table is incorrect length in table file

Number of table entries is not a correct power of 2.

E: Invalid keyword in pair table parameters

Keyword used in list of table parameters is not recognized.

E: Pair table parameters did not set N

List of pair table parameters must include N setting.

E: Pair table cutoffs must all be equal to use with KSpace

When using pair style table with a long-range KSpace solver, the
cutoffs for all atom type pairs must all be the same, since the
long-range solver starts at that cutoff.

*/
