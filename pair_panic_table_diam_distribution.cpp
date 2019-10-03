/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "pair_panic_table_diam_distribution.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

enum{NONE,RLINEAR,RSQ,BMP};

#define MAXLINE  1024
#define EPSILONR 1.0e-6
#define KAPPA 	 240000.0  
#define RCUT     0.88
#define RCUT2    0.7744  // 0.88^2
#define MOT      140.0
#define A        2000.0  // Sticco Cornes 29-Jun 2018
#define B        0.08    // Sticco Cornes 29-Jun 2018 

/* ---------------------------------------------------------------------- */

PairPanicTableDiamDistribution::PairPanicTableDiamDistribution(LAMMPS *lmp) : Pair(lmp)
{
  ntables = 0;
  tables = NULL;
}

/* ---------------------------------------------------------------------- */

PairPanicTableDiamDistribution::~PairPanicTableDiamDistribution()
{
  if (copymode) return;

  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(tabindex);
  }
}

/* ---------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::compute(int eflag, int vflag)
{
  int    i,j,ii,jj,inum,jnum,itype,jtype,itable;
  double xtmp,ytmp,ztmp,vxtmp,vytmp,delx,dely,delz,evdwl,fpair,gpair,delvx,delvy,cexp1;
  double rsq,factor_lj,fraction,value,a,b;
  char   estr[128];
  int    *ilist,*jlist,*numneigh,**firstneigh;
  Table  *tb;
  double granular_factor, socialx, socialy; 
  double force_wall[2];	 			// Sticco 18 Jun 2018 
  double force_desired[2];          // Sticco 18 Jun 2018
  num_pasos++;                              // Sticco Cornes 22 Jun
  union_int_float_t rsq_lookup;
  int tlm1 = tablength - 1;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **v = atom->v;                    // Sticco 18 Jun 2018
  double **f = atom->f;
  double *radius = atom->radius;          // Sticco Cornes 29-Jun 2018
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    vxtmp = v[i][0];      // Ignacio Sticco - Guillermo Frank 17 april 2018
    vytmp = v[i][1];     // Ignacio Sticco - Guillermo Frank 17 april 2018
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {

      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delvx = vxtmp - v[j][0];                            
      delvy = vytmp - v[j][1];   
      rsq = delx*delx + dely*dely;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        tb = &tables[tabindex[itype][jtype]];
        if (rsq < tb->innersq) {
          sprintf(estr,"Pair distance < table inner cutoff: " 
                  "ijtype %d %d dist %g",itype,jtype,sqrt(rsq));
          error->one(FLERR,estr);
        }

        if (tabstyle == LOOKUP) {
          itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
          if (itable >= tlm1) {
            sprintf(estr,"Pair distance > table outer cutoff: " 
                    "ijtype %d %d dist %g",itype,jtype,sqrt(rsq));
            error->one(FLERR,estr);
          }
          fpair = factor_lj * tb->f[itable];
        } else if (tabstyle == LINEAR) {
          itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
          if (itable >= tlm1) {
            sprintf(estr,"Pair distance > table outer cutoff: " 
                    "ijtype %d %d dist %g",itype,jtype,sqrt(rsq));
            error->one(FLERR,estr);
          }
          fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
          value = tb->f[itable] + fraction*tb->df[itable];
          fpair = factor_lj * value;         // Sticco Cornes 29-Jun 2018
          
          value = tb->rsqt[itable] + fraction*tb->drsqt[itable];       // Sticco 18 Jun 2018
          gpair = radius[i]+radius[j]-value;                          // Sticco Cornes 29-Jun 2018

          if (gpair<0.0) gpair=0.0;                                   // Sticco 18 Jun 2018

          /////////////////////////////////////////////////////////////////
          rsq = (radius[i]+radius[j])*(radius[i]+radius[j]);                // Cornes 30 Jun 2018
          itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);   // Cornes 30 Jun 2018
          fraction = (rsq - tb->rsq[itable]) * tb->invdelta;                // Cornes 30 Jun 2018
          value = tb->f[itable] + fraction*tb->df[itable];                  // Cornes 30 Jun 2018
          cexp1 = 1.0/(value * sqrt(rsq));                                  // Cornes 30 Jun 2018
          fpair = A*cexp1* fpair;                                           // Cornes 30 Jun 2018
          /////////////////////////////////////////////////////////////////

        } else if (tabstyle == SPLINE) {
          itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
          if (itable >= tlm1) {
            sprintf(estr,"Pair distance > table outer cutoff: " 
                    "ijtype %d %d dist %g",itype,jtype,sqrt(rsq));
            error->one(FLERR,estr);
          }
          b = (rsq - tb->rsq[itable]) * tb->invdelta;
          a = 1.0 - b;
          value = a * tb->f[itable] + b * tb->f[itable+1] +
            ((a*a*a-a)*tb->f2[itable] + (b*b*b-b)*tb->f2[itable+1]) *
            tb->deltasq6;
          fpair = factor_lj * value;
        } else {
          rsq_lookup.f = rsq;
          itable = rsq_lookup.i & tb->nmask;
          itable >>= tb->nshiftbits;
          fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
          value = tb->f[itable] + fraction*tb->df[itable];
          fpair = factor_lj * value;
        }

        rsq = delx*delx + dely*dely;                                // Sticco 11 April 2019
        granular_factor = KAPPA*gpair*(dely*delvx-delx*delvy)/rsq; // Sticco 18 Jun 2018
        socialx = delx*fpair;                                      // Sticco 18 Jun 2018
        socialy = dely*fpair;                                      // Sticco 18 Jun 2018

        f[i][0] += socialx - granular_factor*dely;                  // Sticco 18 Jun 2018
        f[i][1] += socialy + granular_factor*delx;                  // Sticco 18 Jun 2018
        if (newton_pair || j < nlocal) {      
           f[j][0] -= socialx - granular_factor*dely;              // Sticco 18 Jun 2018      
           f[j][1] -= socialy + granular_factor*delx;              // Sticco 18 Jun 2018
        }

        if (eflag) {
          if (tabstyle == LOOKUP)
            evdwl = tb->e[itable];
          else if (tabstyle == LINEAR || tabstyle == BITMAP)
            evdwl = tb->e[itable] + fraction*tb->de[itable];
          else
            evdwl = a * tb->e[itable] + b * tb->e[itable+1] +
              ((a*a*a-a)*tb->e2[itable] + (b*b*b-b)*tb->e2[itable+1]) *
              tb->deltasq6;
          evdwl *= factor_lj;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
    tb = &tables[tabindex[1][1]];  // Sticco 12 May - (only for one type of atoms!!!)
    compute_wallforce(xtmp, ytmp, vxtmp, vytmp, force_wall, tb, radius[i]);    // Sticco 18 Jun 2018
    f[i][0] += force_wall[0];    // Sticco 18 Jun 2018
    f[i][1] += force_wall[1];    // Sticco 18 Jun 2018
    compute_desiredforce(xtmp, ytmp, vxtmp, vytmp, force_desired,vd,tb); // Sticco 18 Jun 2018
    f[i][0] += force_desired[0];    // Sticco 18 Jun 2018
    f[i][1] += force_desired[1];    // Sticco 18 Jun 2018 

  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::allocate()
{
  allocated = 1;
  const int nt = atom->ntypes + 1;

  memory->create(setflag,nt,nt,"pair:setflag");
  memory->create(cutsq,nt,nt,"pair:cutsq");
  memory->create(tabindex,nt,nt,"pair:tabindex");

  memset(&setflag[0][0],0,nt*nt*sizeof(int));
  memset(&cutsq[0][0],0,nt*nt*sizeof(double));
  memset(&tabindex[0][0],0,nt*nt*sizeof(int));
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::settings(int narg, char **arg)
{

  if (narg < 3) error->all(FLERR,"Illegal pair_style command");

  // new settings

  if (strcmp(arg[0],"lookup") == 0) tabstyle = LOOKUP;
  else if (strcmp(arg[0],"linear") == 0) tabstyle = LINEAR;
  else if (strcmp(arg[0],"spline") == 0) tabstyle = SPLINE;
  else if (strcmp(arg[0],"bitmap") == 0) tabstyle = BITMAP;
  else error->all(FLERR,"Unknown table style in pair_style command");

  tablength = force->inumeric(FLERR,arg[1]);
  if (tablength < 3) error->all(FLERR,"Illegal number of pair table entries");

  // optional keywords
  // assert the tabulation is compatible with a specific long-range solver

  vd = atof(arg[2]);       // Sticco 18 Jun
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"ewald") == 0) ewaldflag = 1;
    else if (strcmp(arg[iarg],"pppm") == 0) pppmflag = 1;
    else if (strcmp(arg[iarg],"msm") == 0) msmflag = 1;
    else if (strcmp(arg[iarg],"dispersion") == 0) dispersionflag = 1;
    else if (strcmp(arg[iarg],"tip4p") == 0) tip4pflag = 1;
    else error->all(FLERR,"Illegal pair_style command");
    iarg++;
  }

  // delete old tables, since cannot just change settings

  for (int m = 0; m < ntables; m++) free_table(&tables[m]);
  memory->sfree(tables);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(tabindex);
  }
  allocated = 0;

  ntables = 0;
  tables = NULL;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::coeff(int narg, char **arg)
{
  if (narg != 4 && narg != 5) error->all(FLERR,"Illegal pair_coeff command");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  int me;
  MPI_Comm_rank(world,&me);
  tables = (Table *)
    memory->srealloc(tables,(ntables+1)*sizeof(Table),"pair:tables");
  Table *tb = &tables[ntables];
  null_table(tb);
  if (me == 0) read_table(tb,arg[2],arg[3]);
  bcast_table(tb);

  // set table cutoff

  if (narg == 5) tb->cut = force->numeric(FLERR,arg[4]);
  else if (tb->rflag) tb->cut = tb->rhi;
  else tb->cut = tb->rfile[tb->ninput-1];

  // error check on table parameters
  // insure cutoff is within table
  // for BITMAP tables, file values can be in non-ascending order

  if (tb->ninput <= 1) error->one(FLERR,"Invalid pair table length");
  double rlo,rhi;
  if (tb->rflag == 0) {
    rlo = tb->rfile[0];
    rhi = tb->rfile[tb->ninput-1];
  } else {
    rlo = tb->rlo;
    rhi = tb->rhi;
  }
  if (tb->cut <= rlo || tb->cut > rhi)
    error->all(FLERR,"Invalid pair table cutoff");
  if (rlo <= 0.0) error->all(FLERR,"Invalid pair table cutoff");

  // match = 1 if don't need to spline read-in tables
  // this is only the case if r values needed by final tables
  //   exactly match r values read from file
  // for tabstyle SPLINE, always need to build spline tables

  tb->match = 0;
  if (tabstyle == LINEAR && tb->ninput == tablength &&
      tb->rflag == RSQ && tb->rhi == tb->cut) tb->match = 1;
  if (tabstyle == BITMAP && tb->ninput == 1 << tablength &&
      tb->rflag == BMP && tb->rhi == tb->cut) tb->match = 1;
  if (tb->rflag == BMP && tb->match == 0)
    error->all(FLERR,"Bitmapped table in file does not match requested table");

  // spline read-in values and compute r,e,f vectors within table

  if (tb->match == 0) spline_table(tb);
  compute_table(tb);

  // store ptr to table in tabindex

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      tabindex[i][j] = ntables;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Illegal pair_coeff command");
  ntables++;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairPanicTableDiamDistribution::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  tabindex[j][i] = tabindex[i][j];

  return tables[tabindex[i][j]].cut;
}

/* ----------------------------------------------------------------------
   read a table section from a tabulated potential file
   only called by proc 0
   this function sets these values in Table:
     ninput,rfile,efile,ffile,rflag,rlo,rhi,fpflag,fplo,fphi,ntablebits
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::read_table(Table *tb, char *file, char *keyword)
{
  char line[MAXLINE];

  // open file

  FILE *fp = force->open_potential(file);
  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open file %s",file);
    error->one(FLERR,str);
  }

  // loop until section found with matching keyword

  while (1) {
    if (fgets(line,MAXLINE,fp) == NULL)
      error->one(FLERR,"Did not find keyword in table file");
    if (strspn(line," \t\n\r") == strlen(line)) continue;     // blank line
    if (line[0] == '#') {wall_extract(tb,line,fp); continue;} // comment
    char *word = strtok(line," \t\n\r");
    if (strcmp(word,keyword) == 0) break;                     // matching keyword
    fgets(line,MAXLINE,fp);                                   // no match, skip section
    param_extract(tb,line);
    fgets(line,MAXLINE,fp);
    for (int i = 0; i < tb->ninput; i++) fgets(line,MAXLINE,fp);
  }

  // read args on 2nd line of section
  // allocate table arrays for file values

  fgets(line,MAXLINE,fp);
  param_extract(tb,line);
  memory->create(tb->rfile,tb->ninput,"pair:rfile");
  memory->create(tb->efile,tb->ninput,"pair:efile");
  memory->create(tb->ffile,tb->ninput,"pair:ffile");

  // setup bitmap parameters for table to read in

  tb->ntablebits = 0;
  int masklo,maskhi,nmask,nshiftbits;
  if (tb->rflag == BMP) {
    while (1 << tb->ntablebits < tb->ninput) tb->ntablebits++;
    if (1 << tb->ntablebits != tb->ninput)
      error->one(FLERR,"Bitmapped table is incorrect length in table file");
    init_bitmap(tb->rlo,tb->rhi,tb->ntablebits,masklo,maskhi,nmask,nshiftbits);
  }

  // read r,e,f table values from file
  // if rflag set, compute r
  // if rflag not set, use r from file

  int itmp;
  double rfile,rnew;
  union_int_float_t rsq_lookup;

  int rerror = 0;
  int cerror = 0;

  fgets(line,MAXLINE,fp);
  for (int i = 0; i < tb->ninput; i++) {
    if (NULL == fgets(line,MAXLINE,fp))
      error->one(FLERR,"Premature end of file in pair table");
    if (4 != sscanf(line,"%d %lg %lg %lg",
                    &itmp,&rfile,&tb->efile[i],&tb->ffile[i]))  ++cerror;

    rnew = rfile;
    if (tb->rflag == RLINEAR)
      rnew = tb->rlo + (tb->rhi - tb->rlo)*i/(tb->ninput-1);
    else if (tb->rflag == RSQ) {
      rnew = tb->rlo*tb->rlo +
        (tb->rhi*tb->rhi - tb->rlo*tb->rlo)*i/(tb->ninput-1);
      rnew = sqrt(rnew);
    } else if (tb->rflag == BMP) {
      rsq_lookup.i = i << nshiftbits;
      rsq_lookup.i |= masklo;
      if (rsq_lookup.f < tb->rlo*tb->rlo) {
        rsq_lookup.i = i << nshiftbits;
        rsq_lookup.i |= maskhi;
      }
      rnew = sqrtf(rsq_lookup.f);
    }

    if (tb->rflag && fabs(rnew-rfile)/rfile > EPSILONR) rerror++;

    tb->rfile[i] = rnew;
  }

  // close file

  fclose(fp);

  // warn if force != dE/dr at any point that is not an inflection point
  // check via secant approximation to dE/dr
  // skip two end points since do not have surrounding secants
  // inflection point is where curvature changes sign

  double r,e,f,rprev,rnext,eprev,enext,fleft,fright;

  int ferror = 0;
  for (int i = 1; i < tb->ninput-1; i++) {
    r = tb->rfile[i];
    rprev = tb->rfile[i-1];
    rnext = tb->rfile[i+1];
    e = tb->efile[i];
    eprev = tb->efile[i-1];
    enext = tb->efile[i+1];
    f = tb->ffile[i];
    fleft = - (e-eprev) / (r-rprev);
    fright = - (enext-e) / (rnext-r);
    if (f < fleft && f < fright) ferror++;
    if (f > fleft && f > fright) ferror++;
  }

  if (ferror) {
    char str[128];
    sprintf(str,"%d of %d force values in table are inconsistent with -dE/dr.\n"
            "  Should only be flagged at inflection points",ferror,tb->ninput);
    error->warning(FLERR,str);
  }

  // warn if re-computed distance values differ from file values

  if (rerror) {
    char str[128];
    sprintf(str,"%d of %d distance values in table with relative error\n"
            "  over %g to re-computed values",rerror,tb->ninput,EPSILONR);
    error->warning(FLERR,str);
  }

  // warn if data was read incompletely, e.g. columns were missing

  if (cerror) {
    char str[128];
    sprintf(str,"%d of %d lines in table were incomplete\n"
            "  or could not be parsed completely",cerror,tb->ninput);
    error->warning(FLERR,str);
  }
}

/* ----------------------------------------------------------------------
   broadcast read-in table info from proc 0 to other procs
   this function communicates these values in Table:
     ninput,rfile,efile,ffile,rflag,rlo,rhi,fpflag,fplo,fphi
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::bcast_table(Table *tb)
{
  MPI_Bcast(&tb->ninput,1,MPI_INT,0,world);
  MPI_Bcast(&tb->wall_ninput,1,MPI_INT,0,world);         // Guillermo Frank - 15 june 2018
  MPI_Bcast(&tb->opening_ninput,1,MPI_INT,0,world);      // Guillermo Frank - 15 june 2018

  int me;
  MPI_Comm_rank(world,&me);
  if (me > 0) {
    memory->create(tb->rfile,tb->ninput,"pair:rfile");
    memory->create(tb->efile,tb->ninput,"pair:efile");
    memory->create(tb->ffile,tb->ninput,"pair:ffile");
    //-----Guillermo Frank - 15 june 2018-------------
    memory->create(tb->w1,tb->wall_ninput,"pair:w1");
    memory->create(tb->w2,tb->wall_ninput,"pair:w2");
    memory->create(tb->w3,tb->wall_ninput,"pair:w3");
    memory->create(tb->w4,tb->wall_ninput,"pair:w4");
    memory->create(tb->w5,tb->wall_ninput,"pair:w5");
    memory->create(tb->c1,tb->wall_ninput,"pair:c1");
    memory->create(tb->c2,tb->wall_ninput,"pair:c2");
    memory->create(tb->c3,tb->wall_ninput,"pair:c3");
    memory->create(tb->c4,tb->wall_ninput,"pair:c4");
    memory->create(tb->c5,tb->wall_ninput,"pair:c5");
    memory->create(tb->c6,tb->wall_ninput,"pair:c6");
    memory->create(tb->c7,tb->wall_ninput,"pair:c7");
    memory->create(tb->c8,tb->wall_ninput,"pair:c8");
    memory->create(tb->c9,tb->wall_ninput,"pair:c9");
    memory->create(tb->op1,tb->opening_ninput,"pair:op1");
    memory->create(tb->op2,tb->opening_ninput,"pair:op2");
    memory->create(tb->op3,tb->opening_ninput,"pair:op3");
    memory->create(tb->op4,tb->opening_ninput,"pair:op4");
    memory->create(tb->op5,tb->opening_ninput,"pair:op5");
    memory->create(tb->op6,tb->opening_ninput,"pair:op6");
    //----------------------------------------------------
  }

  MPI_Bcast(tb->rfile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->efile,tb->ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->ffile,tb->ninput,MPI_DOUBLE,0,world);
  //-----Guillermo Frank - 15 june 2018-------------
  MPI_Bcast(tb->w1,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->w2,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->w3,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->w4,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->w5,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->c1,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->c2,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->c3,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->c4,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->c5,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->c6,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->c7,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->c8,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->c9,tb->wall_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->op1,tb->opening_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->op2,tb->opening_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->op3,tb->opening_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->op4,tb->opening_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->op5,tb->opening_ninput,MPI_DOUBLE,0,world);
  MPI_Bcast(tb->op6,tb->opening_ninput,MPI_DOUBLE,0,world);
  //------------------------------------------------

  MPI_Bcast(&tb->rflag,1,MPI_INT,0,world);
  if (tb->rflag) {
    MPI_Bcast(&tb->rlo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->rhi,1,MPI_DOUBLE,0,world);
  }
  MPI_Bcast(&tb->fpflag,1,MPI_INT,0,world);
  if (tb->fpflag) {
    MPI_Bcast(&tb->fplo,1,MPI_DOUBLE,0,world);
    MPI_Bcast(&tb->fphi,1,MPI_DOUBLE,0,world);
  }
}

/* ----------------------------------------------------------------------
   build spline representation of e,f over entire range of read-in table
   this function sets these values in Table: e2file,f2file
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::spline_table(Table *tb)
{
  memory->create(tb->e2file,tb->ninput,"pair:e2file");
  memory->create(tb->f2file,tb->ninput,"pair:f2file");

  double ep0 = - tb->ffile[0];
  double epn = - tb->ffile[tb->ninput-1];
  spline(tb->rfile,tb->efile,tb->ninput,ep0,epn,tb->e2file);

  if (tb->fpflag == 0) {
    tb->fplo = (tb->ffile[1] - tb->ffile[0]) / (tb->rfile[1] - tb->rfile[0]);
    tb->fphi = (tb->ffile[tb->ninput-1] - tb->ffile[tb->ninput-2]) /
      (tb->rfile[tb->ninput-1] - tb->rfile[tb->ninput-2]);
  }

  double fp0 = tb->fplo;
  double fpn = tb->fphi;
  spline(tb->rfile,tb->ffile,tb->ninput,fp0,fpn,tb->f2file);
}

/* ----------------------------------------------------------------------
   extract attributes from parameter line in table section
   format of line: N value R/RSQ/BITMAP lo hi FPRIME fplo fphi
   N is required, other params are optional
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::param_extract(Table *tb, char *line)
{
  tb->ninput = 0;
  tb->rflag = NONE;
  tb->fpflag = 0;

  char *word = strtok(line," \t\n\r\f");
  while (word) {
    if (strcmp(word,"N") == 0) {
      word = strtok(NULL," \t\n\r\f");
      tb->ninput = atoi(word);
    } else if (strcmp(word,"R") == 0 || strcmp(word,"RSQ") == 0 ||
               strcmp(word,"BITMAP") == 0) {
      if (strcmp(word,"R") == 0) tb->rflag = RLINEAR;
      else if (strcmp(word,"RSQ") == 0) tb->rflag = RSQ;
      else if (strcmp(word,"BITMAP") == 0) tb->rflag = BMP;
      word = strtok(NULL," \t\n\r\f");
      tb->rlo = atof(word);
      word = strtok(NULL," \t\n\r\f");
      tb->rhi = atof(word);
    } else if (strcmp(word,"FPRIME") == 0) {
      tb->fpflag = 1;
      word = strtok(NULL," \t\n\r\f");
      tb->fplo = atof(word);
      word = strtok(NULL," \t\n\r\f");
      tb->fphi = atof(word);
    } else {
      printf("WORD: %s\n",word);
      error->one(FLERR,"Invalid keyword in pair table parameters");
    }
    word = strtok(NULL," \t\n\r\f");
  }

  if (tb->ninput == 0) error->one(FLERR,"Pair table parameters did not set N");
}

/* ----------------------------------------------------------------------
   compute r,e,f vectors from splined values
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::compute_table(Table *tb)
{
  int tlm1 = tablength-1;

  // inner = inner table bound
  // cut = outer table bound
  // delta = table spacing in rsq for N-1 bins

  double inner;
  if (tb->rflag) inner = tb->rlo;
  else inner = tb->rfile[0];
  tb->innersq = inner*inner;
  tb->delta = (tb->cut*tb->cut - tb->innersq) / tlm1;
  tb->invdelta = 1.0/tb->delta;

  // direct lookup tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // e,f = value at midpt of bin
  // e,f are N-1 in length since store 1 value at bin midpt
  // f is converted to f/r when stored in f[i]
  // e,f are never a match to read-in values, always computed via spline interp

  if (tabstyle == LOOKUP) {
    memory->create(tb->e,tlm1,"pair:e");
    memory->create(tb->f,tlm1,"pair:f");

    double r,rsq;
    for (int i = 0; i < tlm1; i++) {
      rsq = tb->innersq + (i+0.5)*tb->delta;
      r = sqrt(rsq);
      tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
      tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
    }
  }

  // linear tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // de,df values = delta from lower edge to upper edge of bin
  // rsq,e,f are N in length so de,df arrays can compute difference
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == LINEAR) {
    memory->create(tb->rsq,tablength,"pair:rsq");
    memory->create(tb->e,tablength,"pair:e");
    memory->create(tb->f,tablength,"pair:f");
    memory->create(tb->de,tlm1,"pair:de");
    memory->create(tb->df,tlm1,"pair:df");
    memory->create(tb->rsqt,tablength,"pair:rsqt");    // Sticco 18 Jun 2018
    memory->create(tb->drsqt,tlm1,"pair:drsqt");       // Sticco 18 Jun 2018


    double r,rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i*tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      tb->rsqt[i] = r;     // Sticco 19 Jun 2018
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i]/r;
      } else {
        tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
      }
    }

    for (int i = 0; i < tlm1; i++) {
      tb->de[i] = tb->e[i+1] - tb->e[i];
      tb->df[i] = tb->f[i+1] - tb->f[i];
      tb->drsqt[i] = tb->rsqt[i+1] - tb->rsqt[i];  // Sticco 18 Jun 2018
    }
  }

  // cubic spline tables
  // N-1 evenly spaced bins in rsq from inner to cut
  // rsq,e,f = value at lower edge of bin
  // e2,f2 = spline coefficient for each bin
  // rsq,e,f,e2,f2 are N in length so have N-1 spline bins
  // f is converted to f/r after e is splined
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == SPLINE) {
    memory->create(tb->rsq,tablength,"pair:rsq");
    memory->create(tb->e,tablength,"pair:e");
    memory->create(tb->f,tablength,"pair:f");
    memory->create(tb->e2,tablength,"pair:e2");
    memory->create(tb->f2,tablength,"pair:f2");

    tb->deltasq6 = tb->delta*tb->delta / 6.0;

    double r,rsq;
    for (int i = 0; i < tablength; i++) {
      rsq = tb->innersq + i*tb->delta;
      r = sqrt(rsq);
      tb->rsq[i] = rsq;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i]/r;
      } else {
        tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r);
      }
    }

    // ep0,epn = dh/dg at inner and at cut
    // h(r) = e(r) and g(r) = r^2
    // dh/dg = (de/dr) / 2r = -f/2r

    double ep0 = - tb->f[0] / (2.0 * sqrt(tb->innersq));
    double epn = - tb->f[tlm1] / (2.0 * tb->cut);
    spline(tb->rsq,tb->e,tablength,ep0,epn,tb->e2);

    // fp0,fpn = dh/dg at inner and at cut
    // h(r) = f(r)/r and g(r) = r^2
    // dh/dg = (1/r df/dr - f/r^2) / 2r
    // dh/dg in secant approx = (f(r2)/r2 - f(r1)/r1) / (g(r2) - g(r1))

    double fp0,fpn;
    double secant_factor = 0.1;
    if (tb->fpflag) fp0 = (tb->fplo/sqrt(tb->innersq) - tb->f[0]/tb->innersq) /
      (2.0 * sqrt(tb->innersq));
    else {
      double rsq1 = tb->innersq;
      double rsq2 = rsq1 + secant_factor*tb->delta;
      fp0 = (splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,sqrt(rsq2)) /
             sqrt(rsq2) - tb->f[0] / sqrt(rsq1)) / (secant_factor*tb->delta);
    }

    if (tb->fpflag && tb->cut == tb->rfile[tb->ninput-1]) fpn =
      (tb->fphi/tb->cut - tb->f[tlm1]/(tb->cut*tb->cut)) / (2.0 * tb->cut);
    else {
      double rsq2 = tb->cut * tb->cut;
      double rsq1 = rsq2 - secant_factor*tb->delta;
      fpn = (tb->f[tlm1] / sqrt(rsq2) -
             splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,sqrt(rsq1)) /
             sqrt(rsq1)) / (secant_factor*tb->delta);
    }

    for (int i = 0; i < tablength; i++) tb->f[i] /= sqrt(tb->rsq[i]);
    spline(tb->rsq,tb->f,tablength,fp0,fpn,tb->f2);
  }

  // bitmapped linear tables
  // 2^N bins from inner to cut, spaced in bitmapped manner
  // f is converted to f/r when stored in f[i]
  // e,f can match read-in values, else compute via spline interp

  if (tabstyle == BITMAP) {
    double r;
    union_int_float_t rsq_lookup;
    int masklo,maskhi;

    // linear lookup tables of length ntable = 2^n
    // stored value = value at lower edge of bin

    init_bitmap(inner,tb->cut,tablength,masklo,maskhi,tb->nmask,tb->nshiftbits);
    int ntable = 1 << tablength;
    int ntablem1 = ntable - 1;

    memory->create(tb->rsq,ntable,"pair:rsq");
    memory->create(tb->e,ntable,"pair:e");
    memory->create(tb->f,ntable,"pair:f");
    memory->create(tb->de,ntable,"pair:de");
    memory->create(tb->df,ntable,"pair:df");
    memory->create(tb->drsq,ntable,"pair:drsq");

    union_int_float_t minrsq_lookup;
    minrsq_lookup.i = 0 << tb->nshiftbits;
    minrsq_lookup.i |= maskhi;

    for (int i = 0; i < ntable; i++) {
      rsq_lookup.i = i << tb->nshiftbits;
      rsq_lookup.i |= masklo;
      if (rsq_lookup.f < tb->innersq) {
        rsq_lookup.i = i << tb->nshiftbits;
        rsq_lookup.i |= maskhi;
      }
      r = sqrtf(rsq_lookup.f);
      tb->rsq[i] = rsq_lookup.f;
      if (tb->match) {
        tb->e[i] = tb->efile[i];
        tb->f[i] = tb->ffile[i]/r;
      } else {
        tb->e[i] = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        tb->f[i] = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
      }
      minrsq_lookup.f = MIN(minrsq_lookup.f,rsq_lookup.f);
    }

    tb->innersq = minrsq_lookup.f;

    for (int i = 0; i < ntablem1; i++) {
      tb->de[i] = tb->e[i+1] - tb->e[i];
      tb->df[i] = tb->f[i+1] - tb->f[i];
      tb->drsq[i] = 1.0/(tb->rsq[i+1] - tb->rsq[i]);
    }

    // get the delta values for the last table entries
    // tables are connected periodically between 0 and ntablem1

    tb->de[ntablem1] = tb->e[0] - tb->e[ntablem1];
    tb->df[ntablem1] = tb->f[0] - tb->f[ntablem1];
    tb->drsq[ntablem1] = 1.0/(tb->rsq[0] - tb->rsq[ntablem1]);

    // get the correct delta values at itablemax
    // smallest r is in bin itablemin
    // largest r is in bin itablemax, which is itablemin-1,
    //   or ntablem1 if itablemin=0

    // deltas at itablemax only needed if corresponding rsq < cut*cut
    // if so, compute deltas between rsq and cut*cut
    //   if tb->match, data at cut*cut is unavailable, so we'll take
    //   deltas at itablemax-1 as a good approximation

    double e_tmp,f_tmp;
    int itablemin = minrsq_lookup.i & tb->nmask;
    itablemin >>= tb->nshiftbits;
    int itablemax = itablemin - 1;
    if (itablemin == 0) itablemax = ntablem1;
    int itablemaxm1 = itablemax - 1;
    if (itablemax == 0) itablemaxm1 = ntablem1;
    rsq_lookup.i = itablemax << tb->nshiftbits;
    rsq_lookup.i |= maskhi;
    if (rsq_lookup.f < tb->cut*tb->cut) {
      if (tb->match) {
        tb->de[itablemax] = tb->de[itablemaxm1];
        tb->df[itablemax] = tb->df[itablemaxm1];
        tb->drsq[itablemax] = tb->drsq[itablemaxm1];
      } else {
            rsq_lookup.f = tb->cut*tb->cut;
        r = sqrtf(rsq_lookup.f);
        e_tmp = splint(tb->rfile,tb->efile,tb->e2file,tb->ninput,r);
        f_tmp = splint(tb->rfile,tb->ffile,tb->f2file,tb->ninput,r)/r;
        tb->de[itablemax] = e_tmp - tb->e[itablemax];
        tb->df[itablemax] = f_tmp - tb->f[itablemax];
        tb->drsq[itablemax] = 1.0/(rsq_lookup.f - tb->rsq[itablemax]);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   set all ptrs in a table to NULL, so can be freed safely
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::null_table(Table *tb)
{
  tb->rfile = tb->efile = tb->ffile = NULL;
  tb->e2file = tb->f2file = NULL;
  tb->rsq = tb->drsq = tb->e = tb->de = NULL;
  tb->f = tb->df = tb->e2 = tb->f2 = NULL;

  //--------Guillermo Frank - 15 june 2018-------------------------
  tb->w1 = tb->w2 = tb->w3 = tb->w4 = tb->w5 = NULL;
  tb->c1 = tb->c2 = tb->c3 = tb->c4 = tb->c5 = NULL;
  tb->c6 = tb->c7 = tb->c8 = tb->c9 = NULL;
  tb->op1 = tb->op2 = tb->op3 = tb->op4 = tb->op5 = tb->op6 = NULL;
  //---------------------------------------------------------------
}

/* ----------------------------------------------------------------------
   free all arrays in a table
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::free_table(Table *tb)
{
  memory->destroy(tb->rfile);
  memory->destroy(tb->efile);
  memory->destroy(tb->ffile);
  memory->destroy(tb->e2file);
  memory->destroy(tb->f2file);

  memory->destroy(tb->rsq);
  memory->destroy(tb->drsq);
  memory->destroy(tb->e);
  memory->destroy(tb->de);
  memory->destroy(tb->f);
  memory->destroy(tb->df);
  memory->destroy(tb->e2);
  memory->destroy(tb->f2);

  //--------Guillermo Frank - 15 june 2018------------
  memory->destroy(tb->w1);
  memory->destroy(tb->w2);
  memory->destroy(tb->w3);
  memory->destroy(tb->w4);
  memory->destroy(tb->w5);
  memory->destroy(tb->c1);
  memory->destroy(tb->c2);
  memory->destroy(tb->c3);
  memory->destroy(tb->c4);
  memory->destroy(tb->c5);
  memory->destroy(tb->c6);
  memory->destroy(tb->c7);
  memory->destroy(tb->c8);
  memory->destroy(tb->c9);
  memory->destroy(tb->op1);
  memory->destroy(tb->op2);
  memory->destroy(tb->op3);
  memory->destroy(tb->op4);
  memory->destroy(tb->op5);
  memory->destroy(tb->op6);
  memory->destroy(tb->rsqt);  	// Sticco 18 Jun 2018
  memory->destroy(tb->drsqt);	// Sticco 18 Jun 2018
  //--------------------------------------------------
}

/* ----------------------------------------------------------------------
   spline and splint routines modified from Numerical Recipes
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::spline(double *x, double *y, int n,
                       double yp1, double ypn, double *y2)
{
  int i,k;
  double p,qn,sig,un;
  double *u = new double[n];

  if (yp1 > 0.99e30) y2[0] = u[0] = 0.0;
  else {
    y2[0] = -0.5;
    u[0] = (3.0/(x[1]-x[0])) * ((y[1]-y[0]) / (x[1]-x[0]) - yp1);
  }
  for (i = 1; i < n-1; i++) {
    sig = (x[i]-x[i-1]) / (x[i+1]-x[i-1]);
    p = sig*y2[i-1] + 2.0;
    y2[i] = (sig-1.0) / p;
    u[i] = (y[i+1]-y[i]) / (x[i+1]-x[i]) - (y[i]-y[i-1]) / (x[i]-x[i-1]);
    u[i] = (6.0*u[i] / (x[i+1]-x[i-1]) - sig*u[i-1]) / p;
  }
  if (ypn > 0.99e30) qn = un = 0.0;
  else {
    qn = 0.5;
    un = (3.0/(x[n-1]-x[n-2])) * (ypn - (y[n-1]-y[n-2]) / (x[n-1]-x[n-2]));
  }
  y2[n-1] = (un-qn*u[n-2]) / (qn*y2[n-2] + 1.0);
  for (k = n-2; k >= 0; k--) y2[k] = y2[k]*y2[k+1] + u[k];

  delete [] u;
}

/* ---------------------------------------------------------------------- */

double PairPanicTableDiamDistribution::splint(double *xa, double *ya, double *y2a, int n, double x)
{
  int klo,khi,k;
  double h,b,a,y;

  klo = 0;
  khi = n-1;
  while (khi-klo > 1) {
    k = (khi+klo) >> 1;
    if (xa[k] > x) khi = k;
    else klo = k;
  }
  h = xa[khi]-xa[klo];
  a = (xa[khi]-x) / h;
  b = (x-xa[klo]) / h;
  y = a*ya[klo] + b*ya[khi] +
    ((a*a*a-a)*y2a[klo] + (b*b*b-b)*y2a[khi]) * (h*h)/6.0;
  return y;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::write_restart(FILE *fp)
{
  write_restart_settings(fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::write_restart_settings(FILE *fp)
{
  fwrite(&tabstyle,sizeof(int),1,fp);
  fwrite(&tablength,sizeof(int),1,fp);
  fwrite(&ewaldflag,sizeof(int),1,fp);
  fwrite(&pppmflag,sizeof(int),1,fp);
  fwrite(&msmflag,sizeof(int),1,fp);
  fwrite(&dispersionflag,sizeof(int),1,fp);
  fwrite(&tip4pflag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairPanicTableDiamDistribution::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&tabstyle,sizeof(int),1,fp);
    fread(&tablength,sizeof(int),1,fp);
    fread(&ewaldflag,sizeof(int),1,fp);
    fread(&pppmflag,sizeof(int),1,fp);
    fread(&msmflag,sizeof(int),1,fp);
    fread(&dispersionflag,sizeof(int),1,fp);
    fread(&tip4pflag,sizeof(int),1,fp);
  }
  MPI_Bcast(&tabstyle,1,MPI_INT,0,world);
  MPI_Bcast(&tablength,1,MPI_INT,0,world);
  MPI_Bcast(&ewaldflag,1,MPI_INT,0,world);
  MPI_Bcast(&pppmflag,1,MPI_INT,0,world);
  MPI_Bcast(&msmflag,1,MPI_INT,0,world);
  MPI_Bcast(&dispersionflag,1,MPI_INT,0,world);
  MPI_Bcast(&tip4pflag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairPanicTableDiamDistribution::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  int itable;
  double fraction,value,a,b,phi;
  int tlm1 = tablength - 1;

  Table *tb = &tables[tabindex[itype][jtype]];
  if (rsq < tb->innersq) error->one(FLERR,"Pair distance < table inner cutoff");

  if (tabstyle == LOOKUP) {
    itable = static_cast<int> ((rsq-tb->innersq) * tb->invdelta);
    if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
    fforce = factor_lj * tb->f[itable];
  } else if (tabstyle == LINEAR) {
    itable = static_cast<int> ((rsq-tb->innersq) * tb->invdelta);
    if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
    value = tb->f[itable] + fraction*tb->df[itable];
    fforce = factor_lj * value;
  } else if (tabstyle == SPLINE) {
    itable = static_cast<int> ((rsq-tb->innersq) * tb->invdelta);
    if (itable >= tlm1) error->one(FLERR,"Pair distance > table outer cutoff");
    b = (rsq - tb->rsq[itable]) * tb->invdelta;
    a = 1.0 - b;
    value = a * tb->f[itable] + b * tb->f[itable+1] +
      ((a*a*a-a)*tb->f2[itable] + (b*b*b-b)*tb->f2[itable+1]) *
      tb->deltasq6;
    fforce = factor_lj * value;
  } else {
    union_int_float_t rsq_lookup;
    rsq_lookup.f = rsq;
    itable = rsq_lookup.i & tb->nmask;
    itable >>= tb->nshiftbits;
    fraction = (rsq_lookup.f - tb->rsq[itable]) * tb->drsq[itable];
    value = tb->f[itable] + fraction*tb->df[itable];
    fforce = factor_lj * value;
  }

  if (tabstyle == LOOKUP)
    phi = tb->e[itable];
  else if (tabstyle == LINEAR || tabstyle == BITMAP)
    phi = tb->e[itable] + fraction*tb->de[itable];
  else
    phi = a * tb->e[itable] + b * tb->e[itable+1] +
      ((a*a*a-a)*tb->e2[itable] + (b*b*b-b)*tb->e2[itable+1]) * tb->deltasq6;
  return factor_lj*phi;
}

/* ----------------------------------------------------------------------
   return the Coulomb cutoff for tabled potentials
   called by KSpace solvers which require that all pairwise cutoffs be the same
   loop over all tables not just those indexed by tabindex[i][j] since
     no way to know which tables are active since pair::init() not yet called
------------------------------------------------------------------------- */

void *PairPanicTableDiamDistribution::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut_coul") != 0) return NULL;
  if (ntables == 0) error->all(FLERR,"All pair coeffs are not set");

  double cut_coul = tables[0].cut;
  for (int m = 1; m < ntables; m++)
    if (tables[m].cut != cut_coul)
      error->all(FLERR,
                 "Pair table cutoffs must all be equal to use with KSpace");
  dim = 0;
  return &tables[0].cut;
}


void PairPanicTableDiamDistribution::wall_extract(Table *tb, char *line,FILE *fp)
{
  int  i,n;
  char word[MAXLINE];
  double o1,o2,o3,o4,o5;
  double wp1,wp2,wp3,wp4,wp5;

  if (sscanf(line+1,"%s %d",word,&n)==2){
         if (strcmp(word,"W") == 0){ 
           tb->wall_ninput = n;
           memory->create(tb->w1,tb->wall_ninput,"pair:w1");
           memory->create(tb->w2,tb->wall_ninput,"pair:w2");
           memory->create(tb->w3,tb->wall_ninput,"pair:w3");
           memory->create(tb->w4,tb->wall_ninput,"pair:w4");
           memory->create(tb->w5,tb->wall_ninput,"pair:w5");
           memory->create(tb->c1,tb->wall_ninput,"pair:c1");
           memory->create(tb->c2,tb->wall_ninput,"pair:c2");
           memory->create(tb->c3,tb->wall_ninput,"pair:c3");
           memory->create(tb->c4,tb->wall_ninput,"pair:c4");
           memory->create(tb->c5,tb->wall_ninput,"pair:c5");
           memory->create(tb->c6,tb->wall_ninput,"pair:c6");
           memory->create(tb->c7,tb->wall_ninput,"pair:c7");
           memory->create(tb->c8,tb->wall_ninput,"pair:c8");
           memory->create(tb->c9,tb->wall_ninput,"pair:c9");

           for (i=0;i<n;i++){
              fgets(word,MAXLINE,fp);
              sscanf(word+1,"%lg %lg %lg %lg %lg",&wp1,&wp2,&wp3,&wp4,&wp5);

              tb->w1[i] = wp1;
              tb->w2[i] = wp2;
              tb->w3[i] = wp3;
              tb->w4[i] = wp4;
              tb->w5[i] = wp5;
              tb->c1[i] = wp4-wp3;
              tb->c2[i] = wp1-wp2;
              tb->c3[i] = wp2*wp3-wp4*wp1;

              tb->c8[i] = 1.0/((tb->c1[i])*(tb->c1[i])+(tb->c2[i])*(tb->c2[i]));
              tb->c4[i] = (tb->c1[i])*(tb->c2[i])*(tb->c8[i]);
        
              tb->c6[i] = 0.0;
              if (tb->c1[i] != 0.0) tb->c6[i] = (tb->c2[i])/(tb->c1[i]);

              tb->c7[i] = (tb->c1[i])*(tb->c1[i])*(tb->c8[i]);

              tb->c5[i] = 0.0;
              tb->c9[i] = 0.0;

              if (tb->c2[i] != 0.0) {
                tb->c5[i] = (tb->c3[i])/(tb->c2[i]); 
                tb->c9[i] = (tb->c1[i])/(tb->c2[i]); 
              }
           } 
         }
         else if (strcmp(word,"O") == 0){
                tb->opening_ninput = n;
                memory->create(tb->op1,tb->opening_ninput,"pair:op1");
                memory->create(tb->op2,tb->opening_ninput,"pair:op2");
                memory->create(tb->op3,tb->opening_ninput,"pair:op3");
                memory->create(tb->op4,tb->opening_ninput,"pair:op4");
                memory->create(tb->op5,tb->opening_ninput,"pair:op5");
                memory->create(tb->op6,tb->opening_ninput,"pair:op6");

                for (i=0;i<n;i++){
                   fgets(word,MAXLINE,fp);
                   sscanf(word+1,"%lg %lg %lg %lg %lg",&o1,&o2,&o3,&o4,&o5);

                   tb->op1[i] = o1;
                   tb->op2[i] = o2;
                   tb->op3[i] = o3;
                   tb->op4[i] = o4;
                   tb->op5[i] = o5;
                   if (o1 == o2) tb->op6[i] = 0.0;        // vertical orientation
                   else if (o3 == o4) tb->op6[i] = 1.0;
                        else  tb->op6[i] = 2.0;           // horizontal orientation
                } 
              }
  }
}


void PairPanicTableDiamDistribution::compute_wallforce(double x, double y, double vx, double vy, double force_wall[], Table *tb, double rad){

  double rsq,numerator,denominator,xp,yp,c1,c2,c3,c4,c5,c6,c7,c8,c9,r,cexp2,cociente,rad2;
  double wp1,wp2,wp3,wp4;

  int inum,jnum,itype,jtype,itable;
  double delx,dely,delz,delvx,delvy,fpair,gpair; 
  double fraction,value;
  char estr[128];
  int tlm1 = tablength - 1;
  int *type = atom->type;
  
  delz=0.0;
  
  force_wall[0] = 0.0;
  force_wall[1] = 0.0;

   
  for (int j = 0; j < tb->wall_ninput; ++j){
    wp1 = tb->w1[j];
    wp2 = tb->w2[j];
    wp3 = tb->w3[j];
    wp4 = tb->w4[j];
    c1 = tb->c1[j];
    c2 = tb->c2[j];
    c3 = tb->c3[j];
    c4 = tb->c4[j];
    c5 = tb->c5[j];
    c6 = tb->c6[j];
    c7 = tb->c7[j];
    c8 = tb->c8[j];
    c9 = tb->c9[j];
        
    if (c1*c2 == 0.0){
    	
        if (c1 == 0.0 && x < wp2 && x > wp1 ){          //pared horizontal
            xp = x;
            yp = wp3;
            delx = x - xp;
            dely = y - yp;          
            rsq = dely*dely;            
            if (rsq < RCUT2) {
                if (tabstyle == LINEAR) {
                    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
                    if (itable >= tlm1) {
                        sprintf(estr,"Pair distance > table outer cutoff: " 
                        "ijtype %d %d dist %g",itype,jtype,sqrt(rsq));
                        error->one(FLERR,estr);
                    }
                    
                    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
                    value = tb->f[itable] + fraction*tb->df[itable];
                    fpair = value;              // Cornes 30-Jun 2018
                    gpair = 0.0;
                    rad2 = rad * rad;                           // Sticco Cornes 29-Jun 2018
                    if (rad2>rsq) gpair = rad-fabs(dely);   // Sticco Cornes 29-Jun 2018

                    /////////////////////////////////////////////////////////////////
                    rsq = rad2;                                                       // Cornes 30 Jun 2018
                    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);   // Cornes 30 Jun 2018
                    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;                // Cornes 30 Jun 2018
                    value = tb->f[itable] + fraction*tb->df[itable];                  // Cornes 30 Jun 2018
                    cexp2 = 1.0/(value * sqrt(rsq));                                  // Cornes 30 Jun 2018
                    fpair = A*cexp2*fpair;                                           // Cornes 30 Jun 2018
                    /////////////////////////////////////////////////////////////////

                    force_wall[0] += -KAPPA*gpair*vx; // Sticco 18 Jun 2018
                    force_wall[1] += dely*fpair;      // Sticco 18 Jun 2018  	
                    
                }	
            }
        
        }

        else if (c2 == 0.0 && y < wp4 && y > wp3 ){     //pared vertical
            xp = wp1;
            yp = y;
            delx = x - xp;
            dely = y - yp;
            rsq = delx*delx;     // Sticco 12 May
            if (rsq < RCUT2) {
                if (tabstyle == LINEAR) {
                    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
                    if (itable >= tlm1) {
                        sprintf(estr,"Pair distance > table outer cutoff: " 
                        "ijtype %d %d dist %g",itype,jtype,sqrt(rsq));
                        error->one(FLERR,estr);
                    }
                    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
                    value = tb->f[itable] + fraction*tb->df[itable];
                    fpair = value;                          // Cornes 30-Jun 2018
                    gpair = 0.0;
                    rad2= rad * rad;                         // Sticco Cornes 29-Jun 2018
                    
                     if (rad2>rsq) gpair = rad-fabs(delx);   // Sticco Cornes 29-Jun 2018
                    /////////////////////////////////////////////////////////////////
                    rsq = rad2;                                                       // Cornes 30 Jun 2018
                    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);   // Cornes 30 Jun 2018
                    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;                // Cornes 30 Jun 2018
                    value = tb->f[itable] + fraction*tb->df[itable];                  // Cornes 30 Jun 2018
                    cexp2 = 1.0/(value * sqrt(rsq));                                  // Cornes 30 Jun 2018
                    fpair = A*cexp2*fpair;                                           // Cornes 30 Jun 2018
                    /////////////////////////////////////////////////////////////////
                    
                    if (rad2>rsq) gpair = rad-fabs(delx);    // Sticco Cornes 29-Jun 2018
                    force_wall[0] += delx*fpair;      // Sticco 18 Jun 2018
                    force_wall[1] += -KAPPA*gpair*vy; // Sticco 18 Jun 2018
                }		
            }		
        }
        
    }

    else{
        numerator = (c1*x + c2*y + c3)*(c1*x + c2*y + c3);
        denominator = c7;   
        rsq = numerator/denominator;
        cociente = rsq;             // Cornes 30 Jun 2018                
        xp = (-c5-y+c6*x)*c4;
        yp = -c9*xp-c5;
        delx = x - xp;
        dely = y - yp; 
        if (rsq < RCUT2) {
            double yup = (-1/(c9))*(x-wp1) + wp3;
            double ydown = (-1/(c9))*(x-wp2) + wp4;
            double yx = (1/(c9))*(x-wp1) + wp3;
            if (y < yup && y > ydown){ 
                if (tabstyle == LINEAR) {
                    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);
                    if (itable >= tlm1) {
                        sprintf(estr,"Pair distance > table outer cutoff: " 
                        "ijtype %d %d dist %g",itype,jtype,sqrt(rsq));
                        error->one(FLERR,estr);
                    }
                    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;
                    value = tb->f[itable] + fraction*tb->df[itable];
                    fpair = value;                // Cornes 29-Jun 2018
                    gpair = 0.0;
                    rad2= rad*rad;                           // Sticco Cornes 29-Jun 2018
                    if (rad2>rsq) gpair = rad-sqrt(rsq);     // Sticco Cornes 29-Jun 2018

                    /////////////////////////////////////////////////////////////////
                    rsq = rad2;                                                       // Cornes 30 Jun 2018
                    itable = static_cast<int> ((rsq - tb->innersq) * tb->invdelta);   // Cornes 30 Jun 2018
                    fraction = (rsq - tb->rsq[itable]) * tb->invdelta;                // Cornes 30 Jun 2018
                    value = tb->f[itable] + fraction*tb->df[itable];                  // Cornes 30 Jun 2018
                    cexp2 = 1.0/(value * sqrt(rsq));                                  // Cornes 30 Jun 2018
                    fpair = A*cexp2*fpair;                                           // Cornes 30 Jun 2018
                    /////////////////////////////////////////////////////////////////
                }
                force_wall[0] += delx*fpair-KAPPA*gpair*(dely*vx-delx*vy)*dely/cociente; // Sticco 18 Jun 2018
                force_wall[1] += dely*fpair+KAPPA*gpair*(dely*vx-delx*vy)*delx/cociente; // Sticco 18 Jun 2018
            }	
        }

    }      
    
  }
    
}




void PairPanicTableDiamDistribution::compute_desiredforce(double x, double y, double vx, double vy, double* force_desired, double vd, Table *tb){
  double xi,xf,yi,yf,rsq,dx,dy,dx_min,dy_min,r,rinv,nx,ny;
  int    pointing,orientation,jmin,ubication,ubication_min;
  double rsqmin = 1000000.0;

  force_desired[0] = 0.0;
  force_desired[1] = 0.0;

    int j=0;

    while(j< tb->opening_ninput){
      xi = tb->op1[j];
      xf = tb->op2[j];
      yi = tb->op3[j];
      yf = tb->op4[j];
      pointing = tb->op5[j];
      orientation = tb->op6[j];

      if(orientation == 0){                    // Vertical orientation
      
          dx = xi - x;
          if (dx*pointing > 0){       //significa que no paso por el opening

            if (y >= yi && y <= yf){
                rsq = dx*dx;
                dy = 0.0;
                ubication = 0;
            }
            else if (y < yi){
                dy = yi - y;
                rsq = dx*dx + dy*dy;      
                ubication = 1;
            }
            else {
                dy = yf - y;
                rsq = dx*dx + dy*dy;       
                ubication = 2;
            }
          }
          else{               //esta en el opening o ya paso
            ubication = 3;
            rsq = 1000000.0;
          }
      }

      else if(orientation == 1){           // Horizontal orientation
          
          dy = yi - y;
          if (dy*pointing < 0){     //No paso por el opening
            if(x >= xi && x <= xf){
                dx = 0.0; 
                rsq = dy*dy;
                ubication = 0;
            }
            else if (xi < x){
                dx = xi - x;
                rsq = dx*dx + dy*dy;
                ubication = 1;
            }
            else{
                dx = xf - x;
                rsq = dx*dx + dy*dy;
                ubication = 2;
            }  
          }
          else{             
            ubication = 3;
            rsq = 1000000.0;    //si ya paso la puerta, la mando al infinito
          }
      }
      
      else{               // Diagonal orientation
          //rsq ==10000; // Falta hacer el caso oblicua
      } 
    
      if (rsq <= rsqmin){
          rsqmin = rsq;
          ubication_min = ubication;
          jmin = j;
          dx_min = dx;
          dy_min = dy;
      }
      j++;
    }

    if(tb->op6[jmin] == 0){    // Vertical orientation
      if (ubication_min == 0){          // Medio del opening
        r = sqrt(rsqmin);
          rinv = 1.0/r;
          nx = dx_min*rinv;
          force_desired[0] = MOT*(vd*nx-vx);   // Sticco 18 Jun 2018
          force_desired[1] = MOT*(vd*ny-vy);   // Sticco 6 Sept 2019
      }
      else if (ubication_min == 3){
          force_desired[0] = 0.0;     // Sticco 18 Jun 2018
          force_desired[1] = 0.0;     // Sticco 18 Jun 2018
      }
      else{
          r = sqrt(rsqmin);
          rinv = 1.0/r;
          nx = dx_min*rinv;
          ny = dy_min*rinv;
          force_desired[0] = MOT*(vd*nx-vx);    // Sticco 18 Jun 2018
          force_desired[1] = MOT*(vd*ny-vy);    // Sticco 18 Jun 2018
      }
    }
    else if(tb->op6[jmin] == 1){    // Horizontal orientation
      if (ubication_min == 0){
        r = sqrt(rsqmin);
          rinv = 1.0/r;
          ny = dy_min*rinv;
          force_desired[0] = MOT*(vd*nx-vx); // Sticco 6 Sept 2019
          force_desired[1] = MOT*(vd*ny-vy);  // Sticco 18 Jun 2018
      }
      else if (ubication_min == 3){
          force_desired[0] = 0.0;     // Sticco 18 Jun 2018
          force_desired[1] = 0.0;     // Sticco 18 Jun 2018
      }
      else{
          r = sqrt(rsqmin);
          rinv = 1.0/r;
          nx = dx_min*rinv;
          ny = dy_min*rinv;
          force_desired[0] = MOT*(vd*nx-vx);  // Sticco 18 Jun 2018
          force_desired[1] = MOT*(vd*ny-vy);  // Sticco 18 Jun 2018
      }
    }
    
}


void PairPanicTableDiamDistribution::search_near_opening(int i, double vector_vd_x[], double vector_vd_y[], Table *tb){


  double xi,xf,yi,yf,rsq,dx,dy,dx_min,dy_min,r,rinv,nx,ny,xtmp,ytmp;
  int    pointing,orientation,jmin,ubication,ubication_min,j;
  double rsqmin = 1000000.0;

  double **x = atom->x;

  xtmp = x[i][0];
  ytmp = x[i][1];

  vector_vd_x[i] = 0.0;
  vector_vd_y[i] = 0.0;

  j=0;

    while(j< tb->opening_ninput){
      xi = tb->op1[j];
      xf = tb->op2[j];
      yi = tb->op3[j];
      yf = tb->op4[j];
      pointing = tb->op5[j];
      orientation = tb->op6[j];

      if(orientation == 0){                    // Vertical orientation
      
          dx = xi - xtmp;
          if (dx*pointing > 0){       //significa que no paso por el opening

            if (ytmp >= yi && ytmp <= yf){
                rsq = dx*dx;
                dy = 0.0;
                ubication = 0;
            }
            else if (ytmp < yi){
                dy = yi - ytmp;
                rsq = dx*dx + dy*dy;      
                ubication = 1;
            }
            else {
                dy = yf - ytmp;
                rsq = dx*dx + dy*dy;       
                ubication = 2;
            }
          }
          else{               //esta en el opening o ya paso
            ubication = 3;
            rsq = 1000000.0;
          }
      }

      else if(orientation == 1){           // Horizontal orientation
          
          dy = yi - ytmp;
          if (dy*pointing < 0){     //No paso por el opening
            if(xtmp >= xi && xtmp <= xf){
                dx = 0.0; 
                rsq = dy*dy;
                ubication = 0;
            }
            else if (xi < xtmp){
                dx = xi - xtmp;
                rsq = dx*dx + dy*dy;
                ubication = 1;
            }
            else{
                dx = xf - xtmp;
                rsq = dx*dx + dy*dy;
                ubication = 2;
            }  
          }
          else{             
            ubication = 3;
            rsq = 1000000.0;    //si ya paso la puerta, la mando al infinito
          }
      }
      
      else{               // Diagonal orientation
          //rsq ==10000; // Falta hacer el caso oblicua
      } 
    
      if (rsq <= rsqmin){
          rsqmin = rsq;
          ubication_min = ubication;
          jmin = j;
          dx_min = dx;
          dy_min = dy;
      }
      j++;
    }

    if(tb->op6[jmin] == 0){    // Vertical orientation
      if (ubication_min == 0){          // Medio del opening
        vector_vd_x[i] = 1.0;
      }
      else if (ubication_min == 3){
        vector_vd_x[i] = 0.0;
        vector_vd_y[i] = 0.0;
      }
      else{
          r = sqrt(rsqmin);
          rinv = 1.0/r;
          nx = dx_min*rinv;
          ny = dy_min*rinv;
          vector_vd_x[i] = nx;
          vector_vd_y[i] = ny;
      }
    }
    else if(tb->op6[jmin] == 1){    // Horizontal orientation
      if (ubication_min == 0){
        vector_vd_y[i] = 1.0;
      }
      else if (ubication_min == 3){
          vector_vd_x[i] = 0.0;
          vector_vd_y[i] = 0.0;
      }
      else{
          r = sqrt(rsqmin);
          rinv = 1.0/r;
          nx = dx_min*rinv;
          ny = dy_min*rinv;
          vector_vd_x[i] = nx;
          vector_vd_y[i] = ny;
      }
    }
    
}