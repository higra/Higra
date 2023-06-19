
#pragma once

// Encapsulation of CSA library and calling program
// https://github.com/rick/CSA
//
// See copy of readme and licence files below
//
// Warning there is a bunch of macro defined here, normally they are all undefined at the end of the file
//
// Author: Benjamin Perret
// Besides wrapper functions, struct lr_arc was modified to hold the index of the edges in the original graph
//


/* README
 CSA uses a cost-scaling push-relabel algorithm to solve assignment
problems. Two basic versions of the source code (with considerable
overlap) are provided: a precise-costs version and a rounded-costs
version. The rounded-costs version was developed to determine whether
working with integer units of epsilon within a scaling iteration (and
hence using only integer arithmetic except to compute the rounded
costs prior to each iteration) would provide a performance gain. On
our machine, it did not.

Although it would be a simple matter to fix this (and we might
make a later version with this change), THE CODE ASSUMES EXISTENCE OF
A PERFECT MATCHING in the input assignment problem. No checks are made
for this condition. If this condition does not hold, the code will run
forever, unless there are fewer nodes on the left-hand side than on
the right (in which case the computed assignment is liable not to be
optimal).

All the code reads the assignment problem in DIMACS format from
standard input and writes the cost of the optimum assignment along
with performance data on the standard output.

Because the code is distributed in the form of a .tar file and some
versions of tar may not know about symbolic links, THE FIRST THING YOU
MUST DO after unpacking the .tar file is check to see whether symbolic
links appear in the prec_costs and round_costs directories. If there
are no symbolic links, type "make links" in both directories. This
action will set up symbolic links to the source code that is in common
between the precise-costs and rounded-costs versions of the program.
Users of systems that don't use symbolic links will have to resort to
something else; a quick-and-dirty fix would be to modify the Makefiles
so they copy instead of making links. Yet another reasonable
alternative is to set the Makefiles up to build hard links (just
remove the "-s" flag from all the "ln" command lines).

The Makefiles are set up to compile many versions of the code, each
with some set of options turned on. The options and the compile-time
macros that activate them are described below:

PREC_COSTS		If PREC_COSTS is defined, edge costs are stored
ROUND_COSTS		precisely as doubles throughout. If
			ROUND_COSTS is defined (as it is guaranteed to
			be if PREC_COSTS isn't), edge costs are stored
			in rounded units of epsilon at each iteration
			so most of the arithmetic done at each stage
			is integer. We deemed the differences between
			these two versions of the code great enough to
			warrant separate source files for some of the
			routines. In particular, csa_types.h, debug.c,
			p_refine.c, p_update.c, refine.c, and
			update_epsilon.c have different source
			versions corresponding to these compile-time
			macros. The best performance we obtained was
			using PREC_COSTS.

MIN_COST		If MIN_COST is defined, the program computes
			the min-cost complete assignment. If MIN_COST
			is undefined, the program computes the
			maximum-cost complete assignment.

SAVE_RESULT		If SAVE_RESULT is defined, the optimum
			assignment computed by the program is saved in
			the file "output.flow" prior to termination.
			The assignment is saved one-line-per-arc,
			according to the format
			"f <lhs id> <rhs id> <cost>".

QUEUE_ORDER		If QUEUE_ORDER is defined, active vertices are
			processed in FIFO order. Otherwise, active
			vertices are processed in stack order.
			QUEUE_ORDER did not seem to improve
			performance for us.

EXPLICIT_LHS_PRICES	If EXPLICIT_LHS_PRICES is defined, a price
			field is stored with each node on the
			left-hand side in the PREC_COSTS case. The
			only place where these prices are used to
			govern the program's behavior in any way is in
			the case where STRONG_PO is defined. Then, the
			computation of the reduced cost for an arc
			uses the stored price for the arc's left
			endpoint rather than inferring an implicit
			price for that node.

	WARNING:	Because explicit lhs prices were added as an
			afterthought, the following statements hold:
			We do not know what effects this use of
			explicit lhs prices has on performance of the
			code with strong price-outs enabled.
			The code for global price updates and price
			refinement do not update prices for lhs nodes
			under any circumstances. The effects of this
			fact on performance are nil, except in the
			STRONG_PO case, where the effects are unknown;
			correctness is unaffected.
			Explicit lhs prices are available only in the
			PREC_COSTS case; I never bothered to implement
			them in the ROUND_COSTS code, although there
			is no especially good reason not to.

USE_PRICE_OUT		USE_PRICE_OUT is relevant only in the
			PREC_COSTS case because pricing out is always
			used to avoid integer overflow in the
			ROUND_COSTS case. If USE_PRICE_OUT is defined,
			arcs are considered once per iteration for
			pricing out. Those arcs with high enough
			reduced cost (where "high enough" depends on
			STRONG_PO as explained below) are moved to
			separate regions of the data structure where
			they no longer contribute to computation
			costs.
STRONG_PO		If STRONG_PO is NOT defined, arcs are priced
			out according to the theoretical bound on the
			reduced cost of an arc whose status may yet
			change during the computation, i.e., if the
			present assignment is epsilon-optimal, arcs
			with reduced cost exceeding 2 * n * epsilon in
			absolute value are fixed. If STRONG_PO IS
			defined, speculative arc fixing is used, i.e.
			the reduced cost limit for priced-in arcs can
			be considerably lower. If the threshold is not
			specified on the command line, a value of 2.0
			* sqrt(n) * sqrt(sqrt(n)) is used. Note that
			lowering this threshold incurs an additional
			cost: since priced-out arcs are no longer
			guaranteed by the theory to obey
			epsilon-optimality, the program periodically
			checks each one to ensure that it remains
			well-behaved. Any arc that doesn't gets moved
			back into the normal arc list, with possible
			modification of its status to preserve
			epsilon-optimality. For some classes of
			problems, STRONG_PO helped performance
			significantly. For others, it hurt.

USE_SP_AUG_FORWARD	These two mutually exclusive macros cause the
USE_SP_AUG_BACKWARD	following behavior: Each execution of
			refine(), upon achieving the condition that
			EXCESS_THRESH units of excess remain, invokes
			an approximate-shortest-path augmentation
			procedure to eliminate the remaining excess.
			If USE_SP_AUG_FORWARD is defined, the search
			for augmenting paths proceeds "forward" from
			each excess, one at a time. If
			USE_SP_AUG_BACKWARD is defined, the search for
			augmenting paths proceeds "backwards" from all
			deficits simultaneously. I found that the
			performance of the forward search so far
			outstripped the backward search's speed that
			it wasn't worth modifying the backward search
			to proceed from a single deficit at a time,
			even though this would probably help its speed
			considerably and is definitely the more
			intelligent approach. Several other minor
			style problems remain with sp_aug_backward.c
			(the code corresponding to
			USE_SP_AUG_BACKWARD), and the code is included
			mainly as a curiosity. Even with its elegance
			improved as much as I know how, it would not
			perform at the level of sp_aug_forward.c
			(which corresponds to USE_SP_AUG_FORWARD), so
			I stopped improving it. Unfortunately for the
			reader who's come this far, even forward
			shortest path augmentation didn't help
			performance on any problem class I tested.
EXCESS_THRESH		Determines the amount of excess at which
			refine() switches to approximate shortest-path
			augmentation. A detestable hack in
			sp_aug_backward.c requires that if
			USE_SP_AUG_BACKWARD is defined, EXCESS_THRESH
			be a constant.
LOG_PATHS		Debugging option that switches on printing of
			augmenting paths used by approximate
			shortest-path augmentation.

QUICK_MIN		Relevant only in the PREC_COSTS case, simply
			because this feature is unimplemented for
			rounded costs. There is no reason in principle
			why implementation of this feature would be
			difficult or ill-advised in the ROUND_COSTS
			case. If QUICK_MIN is defined, attached to
			each node is stored an array of pointers to
			arcs (called the node's best-list) and a
			lower bound (called the node's next-best
			value) on the partial reduced cost of any arc
			not stored in the best-list. During a
			double-push operation, the best-list is
			examined to determine whether at least two of
			its members have partial reduced costs still
			below the next-best value. If they do, these
			two arcs are necessarily the minimum and
			second-minimum reduced-cost arcs for this
			node. If the best-list contains no such arcs,
			the node's next-best value and best-list are
			rebuilt. During a rebuild, the bound used as
			the next-best value depends on the LOOSE_BOUND
			flag. For most problem classes, defining
			QUICK_MIN increased speed considerably in our
			tests.
LOOSE_BOUND		Relevant only in the QUICK_MIN case. If
			LOOSE_BOUND is defined, a node's next-best
			value is set to be the partial reduced cost of
			the costliest arc in the node's new best-list.
			If LOOSE_BOUND is undefined, the next-best
			value is set to be the partial reduced cost of
			the least costly arc excluded from the node's
			new best-list. The latter of these two options
			proved better in our tests, probably because
			all of the node's incident arcs must be
			examined during the rebuild anyway, so nothing
			is gained by storing a looser bound than we
			might.
NUM_BEST		If QUICK_MIN is defined, NUM_BEST is the
			number of candidate arcs to save with each
			node in an attempt to avoid scanning all of a
			node's incident arcs while computing the
			minimum reduced-cost arc for a particular
			node. The larger this number, the greater the
			tendency to avoid looking at all the arcs, but
			the higher the cost of rebuilding each node's
			list of candidates when all arcs must be
			examined. NUM_BEST must be at least 2. If
			tournament sorts could be coded easily for
			general values of NUM_BEST, perhaps we would
			have used a higher value, but our best
			implementation had NUM_BEST = 3.

STORE_REV_ARCS		If STORE_REV_ARCS is defined, fields are
			allocated in the data structures to hold
			reverse arcs, and the parser fills those
			fields. Reverse arcs support global price
			updates and back-arc price-outs. See below.

BACK_PRICE_OUT		Relevant only in the USE_PRICE_OUT case. This
			option is designed to take advantage of the
			fact that if the matching arc incident to a
			node is priced out, all the arcs incident to
			that node may as well be priced out. In the
			STRONG_PO case, this option led to poor
			performance by being too aggressive. With
			STRONG_PO undefined, this option still did not
			seem to help, although all price-outs
			performed in this case are justified by the
			theory. This option is valid only if
			STORE_REV_ARCS is defined.

USE_P_REFINE		If USE_P_REFINE is defined, price refinement
			is switched on. Price refinement determines,
			following the decrease of epsilon at each
			iteration, whether the current assignment is
			already epsilon-optimal before calling refine.
			If the current assignment is already epsilon
			optimal, execution of refine is avoided at
			that iteration. In our tests, refine tended to
			execute faster than the price refinement that
			would have obviated its execution.

USE_P_UPDATE		If USE_P_UPDATE is defined, global price
			updates are switched on. Global price updates
			use what amounts to a scaling shortest-paths
			computation executed at itervals during refine
			to ensure that there is a path from every
			excess to some deficit in the admissible
			graph.
NO_FEAS_PROMISE		If NO_FEAS_PROMISE is defined, p_update()
			and/or sp_aug() check at a certain point
			whether any node with excess lacks an outgoing
			residual arc, and balk if it does. This
			condition cannot arise for the well-behaved
			classes of graphs we tested, and this option
			is normally switched off. To handle
			poorly-behaved graphs properly, somewhat
			greater care than merely defining
			NO_FEAS_PROMISE would be required.
P_U_ZERO_BACK_MCH_ARCS	If P_U_ZERO_BACK_MCH_ARCS is defined,
			p_update() preprocesses the graph at each
			invocation to reduce the number of arithmetic
			operations required during node scans. I never
			found a class of graphs for which this
			preprocessing paid off, so this option is
			normally (and best) left undefined.

DEBUG			If DEBUG is defined, various forms of highly
			verbose debugging output are produced. Not for
			the faint of heart.
CHECK_EPS_OPT		If CHECK_EPS_OPT is defined, epsilon
			optimality of the current assignment is
			verified at critical stages of the
			computation, and error messages are printed if
			the check fails. This option was used for
			debugging.
VERBOSE_TIME		If VERBOSE_TIME is defined, timing information
			for each iteration is displayed separately.
			Otherwise, only aggregate time is displayed
			upon completion.

Command line usage:
progname <scale> <up_freq> <po_thresh> <po_ck_freq>
<scale>		scale factor
<up_freq>	global update frequency in terms of relabelings
<po_thresh>	reduced-cost threshold in units of epsilon for strong
		price-outs
<po_ck_freq>	frequency in terms of relabelings of checking
		priced-out arcs for pricing back in
All command line parameters are optional and have reasonable default
values. If global updates are not used, no <up_freq> parameter is
expected, nor may one be supplied. If strong price-outs are not used,
no <po_thresh> or <po_ck_freq> parameters are expected, nor may either
be supplied. Supplying any parameter on the command line requires that
all those preceding it in the above description be supplied also.

The technical report describing the research surrounding this code is
available in compressed postscript by anonymous ftp from
theory.stanford.edu; the filename is pub/goldberg/stan-cs-93-1481.ps.Z.
The code we called CSA-B in our paper is prec_costs/csa_s.
The code we called CSA-Q in our paper is prec_costs/csa_s_qm.
The code we called CSA-S in our paper is prec_costs/csa_s_spo.

Please report any problems to Robert Kennedy, robert@cs.stanford.edu.
No one makes any commitment to fix problems with this code, although
we probably will try to address complaints about it. Please see the
copyright notice in main.c.
 */

/* Copyright of the CSA library by Andrew Goldberg and Robert Kennedy
COPYRIGHT C 1994 by Andrew Goldberg and Robert Kennedy.
Permission to use free of charge is granted provided
that proper acknowledgments are given.

This software comes with NO WARRANTY, expressed or implied. By way
of example, but not limitation, we make no representations of warranties
of merchantability or fitness for any particular purpose or that the use
of the software components or documentation will not infringe any patents,
copyrights, trademarks, or other rights.
 */
/* Original header from main.c calling CSA library by Robert Kennedy
This program solves assignment problems posed in DIMACS format.
Written by Robert Kennedy
	   Department of Computer Science
	   Building 460
	   Stanford University
	   Stanford, CA 94305-2140
	   USA
in cooperation with Andrew V. Goldberg.
Revision history:
July, 1994	Option for (approximate) shortest-augmenting path
		finish-up stage (either "forward" or "backward" search
		added, along with a year's minor changes and speedups.
July, 1993	First distribution.
Copyright 1993, 1994 by Robert Kennedy
Permission to copy, exerpt, or prepare derivative works from this code
is granted provided this notice is included.
To obtain a copy of this program, send electronic mail to
ftp-request@theory.stanford.edu with the subject line "send csas.tar";
you will be mailed a uuencoded .tar file.
Please report any problems to robert@cs.stanford.edu.
*/




#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../../graph.hpp"

#ifdef __APPLE__
#include <malloc/malloc.h>
#else

#include <malloc.h>

#endif

//#include <sys/types.h>
//#include <sys/times.h>


namespace hg {
    namespace graph_algorithms {


// NOTE: This code has been tested with the following options only:
// It won't probably work with other options as struct lr_arc was modified (edge_index field added),
// this would require slight modifications in the code path that uses this struct with the other options (mainly arc initialization and copies I think).
#define QUICK_MIN
#define MIN_COST

/** csa_types.h **/

#define    PREC_COSTS

#if    defined(QUICK_MIN) && !defined(NUM_BEST)
#define    NUM_BEST    3
#endif

#if    defined(USE_SP_AUG_FORWARD) || defined(USE_SP_AUG_BACKWARD)
#ifndef	USE_SP_AUG
#define	USE_SP_AUG
#endif
#endif

#if    defined(USE_P_UPDATE) || defined(BACK_PRICE_OUT) || \
    defined(USE_SP_AUG_BACKWARD)
#define	STORE_REV_ARCS
#endif

        typedef struct lhs_node {
#if    defined(QUICK_MIN)
            struct {
                /*
                flag used to indicate to
                double_push() that so few arcs
                are incident that best[] is
                useless.
                */
#ifdef    QUICK_MIN
                size_t few_arcs: 1;
#endif
            } node_info;
#ifdef    QUICK_MIN
            /*
            list of arcs to consider first in
            calculating the minimum-reduced-cost
            incident arc; if we find it here, we
            need look no further.
            */
            struct lr_arc *best[NUM_BEST];
            /*
            bound on the reduced cost of an arc we
            can be certain still belongs among
            those in best[].
            */
            double next_best;
#endif
#endif
#ifdef    EXPLICIT_LHS_PRICES
            /*
                        price of this node.
                        */
                        double	p;
#endif
            /*
            first arc in the arc array associated
            with this node.
            */
            struct lr_arc *priced_out;
            /*
            first priced-in arc in the arc array
            associated with this node.
            */
            struct lr_arc *first;
            /*
            matching arc (if any) associated with
            this node; NULL if this node is
            unmatched.
            */
            struct lr_arc *matched;
#if    defined(USE_P_UPDATE)
            /*
                        price change required on this node (in
                        units of epsilon) to ensure that its
                        excess can reach a deficit in the
                        admissible graph. computed and used in
                        p_update().
                        */
                        int64_t	delta_reqd;
#endif
#ifdef    USE_SP_AUG_BACKWARD
            struct	lr_arc	*aug_path;
#endif
        } *lhs_ptr;

        typedef struct rhs_node {
            struct {
#ifdef    USE_P_REFINE
                /*
                            depth-first search flags.
                            dfs is to determine whether
                            admissible graph contains a
                            cycle in p_refine().
                            */
                            size_t	srchng : 1;
                            size_t	srched : 1;
#endif
                /*
                flag to indicate this node's
                matching arc (if any) is
                priced in.
                */
                size_t priced_in: 1;
            } node_info;
            /*
            lhs node this rhs node is matched to.
            */
            lhs_ptr matched;
            /*
            price of this node.
            */
            double p;
#ifdef    USE_SP_AUG_FORWARD
            struct	lr_arc	*aug_path;
#endif
#if    defined(USE_P_REFINE) || defined(USE_P_UPDATE) || defined(USE_SP_AUG)
            /*
                        number of epsilons of price change
                        required at this node to accomplish
                        p_refine()'s or p_update()'s goal.
                        */
                        int64_t	key;
                        /*
                        fields to maintain buckets of nodes as
                        lists in p_refine() and p_update().
                        */
                        struct	rhs_node	*prev, *next;
#endif
#ifdef    STORE_REV_ARCS
            /*
                        first back arc in the arc array
                        associated with this node.
                        */
                        struct	rl_arc	*priced_out;
                        /*
                        first priced-in back arc in the arc
                        array associated with this node.
                        */
                        struct	rl_arc	*back_arcs;
#endif
        } *rhs_ptr;

#ifdef    STORE_REV_ARCS
        typedef	struct	rl_arc		{
                        /*
                        lhs node associated with this back
                        arc. some would have liked the name
                        head better.
                        */
                        lhs_ptr	tail;
#if	defined(USE_P_UPDATE) || defined(USE_SP_AUG_BACKWARD)
                        /*
                        cost of this back arc. this cost gets
                        modified to incorporate other arc
                        costs in p_update() and sp_aug(),
                        while forward arc costs remain
                        constant throughout.
                        */
                        double	c;
#endif
#if	defined(USE_PRICE_OUT) || defined(USE_SP_AUG_BACKWARD)
                        /*
                        this arc's reverse in the forward arc
                        list.
                        */
                        struct	lr_arc	*rev;
#endif
                        }	*rl_aptr;
#endif

        typedef struct lr_arc {
            /*
            rhs node associated with this arc.
            */
            rhs_ptr head;
            /*
            arc cost.
            */
            double c;

            index_t edge_index;
#ifdef    USE_SP_AUG_FORWARD
            lhs_ptr	tail;
#endif
#ifdef    STORE_REV_ARCS
            /*
                        this arc's reverse in the back arc
                        list.
                        */
                        struct	rl_arc	*rev;
#endif
        } *lr_aptr;

        typedef struct stack_st {
            /*
            Sometimes stacks have lhs nodes, and
            other times they have rhs nodes. So
            there's a little type clash;
            everything gets cast to (char *) so we
            can use the same structure for both.
            */
            char **bottom;
            char **top;
        } *stack;

        typedef struct queue_st {
            /*
            Sometimes queues have lhs nodes, and
            other times they have rhs nodes. So
            there's a little type clash;
            everything gets cast to (char *) so we
            can use the same structure for both.
            */
            char **head;
            char **tail;
            char **storage;
            char **end;
            size_t max_size;
        } *queue;


/** END csa_types.h **/

/** csa_defs.h **/

#define    TRUE    1
#define    FALSE    0
#define    MAXLINE    100
#define    DEFAULT_SCALE_FACTOR    10
#define    DEFAULT_PO_COST_THRESH    (2.0 * sqrt((double) n) * \
                 sqrt(sqrt((double) n)))
#define    DEFAULT_PO_WORK_THRESH    50
#define    DEFAULT_UPD_FAC        2
#if    defined(USE_SP_AUG_FORWARD) || defined(USE_SP_AUG_BACKWARD)
#ifndef	USE_SP_AUG
#define	USE_SP_AUG
#endif
#endif

#ifdef    USE_SP_AUG
#define	EXCESS_THRESH	127
#else
#define    EXCESS_THRESH    0
#endif

#if    defined(USE_P_UPDATE) || defined(STRONG_PO)
#define	WORK_TYPE	size_t
#define	REFINE_WORK	relabelings
#endif

#if    defined(DEBUG) && defined(ROUND_COSTS)
#define	MAGIC_MARKER	0xAAAAAAAA
#endif

#ifdef    QUEUE_ORDER
#define	ACTIVE_TYPE	queue
#define	create_active(size)	active = q_create(size)
#define	make_active(v)		enq(active, (char *) v)
#define	get_active_node(v)	v = (lhs_ptr) deq(active)
#else
#define    ACTIVE_TYPE    stack
#define    create_active(size)    active = st_create(size)
#define    make_active(v)        st_push(active, (char *) v)
#define    get_active_node(v)    v = (lhs_ptr) st_pop(active)
#endif

#define    st_push(s, el) \
{\
*(s->top) = (char *) el;\
s->top++;\
}

#define    st_empty(s)    (s->top == s->bottom)

#define    enq(q, el) \
{\
*(q->tail) = el;\
if (q->tail == q->end) q->tail = q->storage;\
else q->tail++;\
}

#define q_empty(q) (q->head == q->tail ? 1 : 0)

#define    insert_list(node, head) \
{\
node->next = (*(head));\
(*(head))->prev = node;\
(*(head)) = node;\
node->prev = tail_rhs_node;\
}

#define    delete_list(node, head) \
{\
if (node->prev == tail_rhs_node)\
  (*(head)) = node->next;\
node->prev->next = node->next;\
node->next->prev = node->prev;\
}

/*
The author hereby apologizes for the following incomprehensible
muddle. Price-outs involve moving arcs around in the data structure,
and it turns out to be faster to copy them field-by-field than to use
memcpy() because they're so small. But the set of fields an arc has
depends on lots of things, hence this mess.
*/

#if    defined(USE_PRICE_OUT) || defined(ROUND_COSTS)
#ifdef	STORE_REV_ARCS
#ifdef	ROUND_COSTS
#define	copy_lr_arc(a, b) \
{\
b->head = a->head;\
b->c_init = a->c_init;\
b->c = a->c;\
b->rev = a->rev;\
}
#else	/* ROUND_COSTS */
#define	copy_lr_arc(a, b) \
{\
b->head = a->head;\
b->c = a->c;\
b->rev = a->rev;\
}
#endif	/* ROUND_COSTS */

#ifdef	USE_P_UPDATE
#define	copy_rl_arc(a, b) \
    { b->tail = a->tail; b->c = a->c; b->rev = a->rev; }
#else	/* USE_P_UPDATE */
#define	copy_rl_arc(a, b) \
    { b->tail = a->tail; b->rev = a->rev; }
#endif	/* USE_P_UPDATE */

#define	exch_rl_arcs(a, b) \
{\
copy_rl_arc(b, tail_rl_arc);\
copy_rl_arc(a, b);\
copy_rl_arc(tail_rl_arc, a);\
}
#else	/* STORE_REV_ARCS */
#ifdef	PREC_COSTS
#define	copy_lr_arc(a, b) \
{\
b->head = a->head;\
b->c = a->c;\
}
#else	/* PREC_COSTS */
#define	copy_lr_arc(a, b) \
{\
b->head = a->head;\
b->c_init = a->c_init;\
b->c = a->c;\
}
#endif	/* PREC_COSTS */
#endif	/* STORE_REV_ARCS */

#define	exch_lr_arcs(a, b) \
{\
copy_lr_arc(b, tail_lr_arc);\
copy_lr_arc(a, b);\
copy_lr_arc(tail_lr_arc, a);\
}

        extern	lr_aptr	tail_lr_arc;
#ifdef	STORE_REV_ARCS
        extern	rl_aptr	tail_rl_arc;
#endif

#ifdef	STORE_REV_ARCS
#define	price_in_rev(a) \
{ \
register	rl_aptr	b_a = --a->head->back_arcs; \
register	rl_aptr	a_r = a->rev; \
if (b_a != a_r) \
  { \
  register	lr_aptr	b_r = b_a->rev; \
  exch_rl_arcs(b_a, a_r); \
  b_r->rev = a_r; \
  a->rev = b_a; \
  } \
}

#define	price_out_rev(a) \
{ \
register	rl_aptr	b_a = a->head->back_arcs; \
register	rl_aptr	a_r = a->rev; \
if (b_a != a_r) \
  { \
  register	lr_aptr	b_r = b_a->rev; \
  exch_rl_arcs(b_a, a_r); \
  b_r->rev = a_r; \
  a->rev = b_a; \
  } \
a->head->back_arcs++; \
}

#define	handle_rev_pointers(a, b)	{ a->rev->rev = b; b->rev->rev = a; }
#else	/* STORE_REV_ARCS */
#define	price_in_rev(a)		/* do nothing */
#define	price_out_rev(a)	/* do nothing */
#define	handle_rev_pointers(a, b)	/* do nothing */
#endif	/* STORE_REV_ARCS */

#define	price_in_unm_arc(v, a) \
{ \
register	lr_aptr	f_a = --v->first; \
price_in_rev(a); \
if (f_a != a) \
  { \
  if (v->matched == f_a) v->matched = a; \
  handle_rev_pointers(a, f_a); \
  exch_lr_arcs(a, f_a); \
  } \
}

#define	price_in_mch_arc(v, a) \
{ \
register	lr_aptr	f_a = --v->first; \
price_in_rev(a); \
a->head->node_info.priced_in = TRUE; \
if (f_a != a) \
  { \
  v->matched = f_a; \
  handle_rev_pointers(a, f_a); \
  exch_lr_arcs(a, f_a); \
  } \
}

#define	price_out_unm_arc(v, a) \
{ \
register	lr_aptr	f_a = v->first++; \
price_out_rev(a); \
if (f_a != a) \
  { \
  if (v->matched == f_a) v->matched = a; \
  handle_rev_pointers(a, f_a); \
  exch_lr_arcs(a, f_a); \
  } \
}

#define	price_out_mch_arc(v, a) \
{ \
register	lr_aptr	f_a = v->first++; \
price_out_rev(a); \
a->head->node_info.priced_in = FALSE; \
if (f_a != a) \
  { \
  v->matched = f_a; \
  handle_rev_pointers(a, f_a); \
  exch_lr_arcs(a, f_a); \
  } \
}
#endif    /* USE_PRICE_OUT || ROUND_COSTS */


/** END csa_defs.h **/


        class CSA {
        public:

            // construct and solve assignment problem
            template<typename T1, typename T2>
            CSA(const xt::xexpression<T1> &xsources,
                const xt::xexpression<T1> &xtargets,
                index_t num_vertices,
                const xt::xexpression<T2> &xweights) {
                auto &sources = xsources.derived_cast();
                auto &targets = xtargets.derived_cast();
                auto &weights = xweights.derived_cast();
                hg_assert_integral_value_type(weights);
                hg_assert_1d_array(sources);
                hg_assert_1d_array(targets);
                hg_assert_integral_value_type(sources);
                hg_assert_integral_value_type(targets);
                hg_assert_same_shape(sources, targets);
                hg_assert_same_shape(sources, weights);
                hg_assert_1d_array(weights);
                size_t n = num_vertices;
                size_t m = sources.size();
                hg_assert(n > 0, "Graph cannot be empty.");
                hg_assert(m > 0, "Graph must have a least one edge.");
                hg_assert((n % 2) == 0, "Number of vertices must be even.");
                _init(n, m);
                init(sources, targets, weights);
                main();
            }

            // destructor
            ~CSA() {
                _delete();
            }

            // number of edges in the assignment (n)
            index_t edges() { return result_n; }

            // total cost of the assignment
            index_t cost() { return result_cost; }

            array_1d<index_t> edge_indices() {
                std::vector<size_t> shape = {(size_t) result_n};
                return xt::adapt(&result_edge_index[0], shape);
            }


            // Mostly a copy/past of original library c files, globals become instance members
            // It's probably best for your health that you do not look
            // at anything in this file beyond this point.

        private:

            // variables to hold result of assignment
            index_t result_n;
            index_t *result_edge_index;
            index_t result_cost;

            void _init(index_t _n, index_t _m) {
                result_n = 0;
                result_edge_index = NULL;
                result_cost = 0;

                this->n = (size_t)_n;
                this->m = (size_t)_m;
                head_lhs_node = NULL, tail_lhs_node = NULL;
                head_rhs_node = NULL, tail_rhs_node = NULL;
                head_lr_arc = NULL, tail_lr_arc = NULL;
#ifdef    STORE_REV_ARCS
                head_rl_arc=NULL, tail_rl_arc=NULL;
#endif
                double_pushes = 0,
                pushes = 0,
                relabelings = 0,
                refines = 0
                /*refine_time = 0*/;
#ifdef    USE_P_REFINE
                p_refines = 0,
                      r_scans = 0,
                /*p_refine_time = 0*/;
#endif
#ifdef    USE_P_UPDATE
                p_updates = 0,
                      u_scans = 0,
                p_update_time = 0;
#endif
#ifdef    USE_SP_AUG
                sp_augs = 0,
                a_scans = 0,
            sp_aug_time = 0;
#endif
#ifdef    STRONG_PO
                fix_ins = 0;
#endif
#ifdef    QUICK_MIN
                rebuilds = 0,
                scans = 0,
                non_scans = 0;
#endif
                po_cost_thresh = 0;
                scale_factor = 0;    /* scaling factor */
#ifdef    USE_P_UPDATE
                upd_work_thresh=0;/* work threshhold for global update */
#endif
#ifdef    STRONG_PO
#ifdef	ROUND_COSTS
                banish_thresh=0;
#endif
                po_work_thresh=0;	/* work threshhold for price-in checks */
#endif
                epsilon = 0;    /* scaling parameter */
                min_epsilon = 0;    /* snap to this value when epsilon small */
                total_e = 0;    /* total excess */
                active = NULL;        /* list of active nodes */
#ifdef    USE_P_REFINE
                reached_nodes=NULL; /* nodes reached in topological ordering */
#endif
#if    defined(USE_P_REFINE) || defined(USE_P_UPDATE) || defined(USE_SP_AUG)
                bucket=NULL;	/* buckets for use in price refinements */
                num_buckets=0;	/* number of buckets */
#endif
            }

            void _delete() {
                if (result_edge_index != NULL) { delete[] result_edge_index; }

#if    defined(USE_P_REFINE) || defined(USE_P_UPDATE) || defined(USE_SP_AUG)
                free(bucket);
#endif
                free(head_lr_arc);
#ifdef    STORE_REV_ARCS
                free(head_rl_arc);
#endif
                free(head_lhs_node);
                free(head_rhs_node);
#ifdef    USE_P_REFINE
                st_destroy(reached_nodes);
#endif
#ifdef  QUEUE_ORDER
                q_destroy(active);
#else
                st_destroy(active);
#endif
            }

///////////////////////////////////////////////////////////////////////////
// main.c /////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

/* ------------------------- Problem size variables -------------------- */
            size_t n, m;

/* --------------- Data structures describing the problem -------------- */
            lhs_ptr head_lhs_node, tail_lhs_node;
            rhs_ptr head_rhs_node, tail_rhs_node;
            lr_aptr head_lr_arc, tail_lr_arc;
#ifdef    STORE_REV_ARCS
            rl_aptr	head_rl_arc, tail_rl_arc;
#endif

/* ------------------- Bookkeeping/profiling variables ----------------- */
            size_t double_pushes,
                    pushes,
                    relabelings,
                    refines
                    /*refine_time*/;
#ifdef    USE_P_REFINE
            size_t	p_refines,
                    r_scans,
                    /*p_refine_time*/;
#endif
#ifdef    USE_P_UPDATE
            size_t	p_updates,
                    u_scans,
                    p_update_time;
#endif
#ifdef    USE_SP_AUG
            size_t	sp_augs,
                    a_scans,
                    sp_aug_time;
#endif
#ifdef    STRONG_PO
            size_t	fix_ins;
#endif
#ifdef    QUICK_MIN
            size_t rebuilds,
                    scans,
                    non_scans;
#endif

/* ------------------------- Tunable variables ------------------------- */
/*
Cost threshold for pricing out: used even when price-outs are
switched off to make bounding reduced-cost differences easier in
double_push(). In principle this is not necessary, but it makes the
code better.
*/
            double po_cost_thresh;
            double scale_factor;    /* scaling factor */
#ifdef    USE_P_UPDATE
            WORK_TYPE	upd_work_thresh;/* work threshhold for global update */
#endif
#ifdef    STRONG_PO
#ifdef	ROUND_COSTS
            /*
            Cost threshhold for certainty that an arc will never be priced in,
            when strong price-outs are used. We need this because the possibility
            of pricing an arc in requires us to maintain the rounded cost for that
            arc in units of epsilon, but doing so for arcs with high reduced cost
            generates integer overflows. Hence only those priced-out arcs that
            aren't banished entirely have their rounded reduced costs calculated.
            */
            double	banish_thresh;
#endif
            WORK_TYPE	po_work_thresh;	/* work threshhold for price-in checks */
#endif

/*
Processing variables.
*/
            double epsilon;    /* scaling parameter */
            double min_epsilon;    /* snap to this value when epsilon small */
            size_t total_e;    /* total excess */
            ACTIVE_TYPE active;        /* list of active nodes */
#ifdef    USE_P_REFINE
            stack		reached_nodes; /* nodes reached in topological ordering */
#endif
#if    defined(USE_P_REFINE) || defined(USE_P_UPDATE) || defined(USE_SP_AUG)
            rhs_ptr		*bucket;	/* buckets for use in price refinements */
            int64_t		num_buckets;	/* number of buckets */
#endif

/*
Miscellaneous variables.
*/



            /*
            void describe_self() {
                static char *desc[20];
                int i = 0;
        #ifdef    QUICK_MIN
                char minstr[40];
        #endif

        #ifdef    ROUND_COSTS
                desc[i++] = "Rounded costs";
        #endif
        #ifdef    PREC_COSTS
                desc[i++] = "Precise costs";
        #ifdef    USE_PRICE_OUT
                desc[i++] = "Price-outs";
        #endif
        #endif
        #ifdef    STRONG_PO
                desc[i++] = "Strong price-outs";
        #endif
        #ifdef    BACK_PRICE_OUT
                desc[i++] = "Back price-outs";
        #endif
        #ifdef    USE_P_UPDATE
                desc[i++] = "Global updates";
        #endif
        #ifdef    USE_SP_AUG_FORWARD
                desc[i++] = "Forward SAP cleanup";
        #endif
        #ifdef    USE_SP_AUG_BACKWARD
                desc[i++] = "Backward SAP cleanup";
        #endif
        #ifdef    STORE_REV_ARCS
                desc[i++] = "Reverse arcs";
        #endif
        #ifdef    USE_P_REFINE
                desc[i++] = "Price refinement";
        #endif
        #ifdef    QUEUE_ORDER
                desc[i++] = "Queue ordering";
        #else
                desc[i++] = "Stack ordering";
        #endif
        #ifdef    QUICK_MIN
                (void) sprintf(minstr, "Quick minima; NUM_BEST = %d", NUM_BEST);
                desc[i++] = minstr;
        #endif

                desc[i] = NULL;
        #if 0
                (void) fprintf(stderr,"CSA: ");
              for (i = 5; i > 0; i--)
                (void) fprintf(stderr,"=");

              (void) fprintf(stderr," %s", desc[0]);
              for (i = 1; desc[i]; i++)
                (void) fprintf(stderr,"; %s", desc[i]);
              (void) fprintf(stderr," ");
              for (i = 5; i > 0; i--)
                (void) fprintf(stderr,"=");
              (void) fprintf(stderr,"\n");
        #endif // 0
            }
        */

            template<typename T1, typename T2>
            void init(const xt::xexpression<T1> &xsources,
                      const xt::xexpression<T1> &xtargets,
                      const xt::xexpression<T2> &xweights) {
                auto &sources = xsources.derived_cast();
                auto &targets = xtargets.derived_cast();
                auto &weights = xweights.derived_cast();
#if    defined(USE_P_REFINE) || defined(USE_P_UPDATE) || defined(USE_SP_AUG)
                rhs_ptr	r_v;
                int64_t	i;
#endif
#ifdef    QUICK_MIN
                lhs_ptr l_v;
#endif

                //describe_self();
                epsilon = (double)parse(sources, targets, weights);

                scale_factor = DEFAULT_SCALE_FACTOR;

#ifdef    USE_P_UPDATE
                upd_work_thresh = DEFAULT_UPD_FAC * n;
#endif

#ifdef    STRONG_PO
                po_cost_thresh = DEFAULT_PO_COST_THRESH;
#ifdef	ROUND_COSTS
          banish_thresh = 10.0 * (double) n * (scale_factor + 1);
#endif

            po_work_thresh = DEFAULT_PO_WORK_THRESH * n;
#else
                po_cost_thresh = 2.0 * (double) n * (scale_factor + 1);
#endif


                create_active(n);
#ifdef    USE_P_REFINE
                reached_nodes = st_create(n);
#endif
#if    defined(USE_P_REFINE) || defined(USE_P_UPDATE) || defined(USE_SP_AUG)
#ifdef	PREC_COSTS
                num_buckets = scale_factor * n + 1;
#else
                num_buckets = 2 * scale_factor * n + 1;
#endif
                bucket = (rhs_ptr *) malloc((size_t) num_buckets * sizeof(rhs_ptr));
                if (bucket == NULL)
                  {
                  (void) printf("Insufficient memory.\n");
                  exit(9);
                  }
                for (i = 0; i < num_buckets; i++)
                  bucket[i] = tail_rhs_node;
                for (r_v = head_rhs_node; r_v != tail_rhs_node; r_v++)
                  r_v->key = num_buckets;
#endif
#ifdef    QUICK_MIN
                for (l_v = head_lhs_node; l_v != tail_lhs_node; l_v++)
                    if (!l_v->node_info.few_arcs)
                        best_build(l_v);
/*
Count only those builds that take place after initialization; first
setup is free.
*/
                rebuilds = 0;
#endif
            }

            double compute_cost() {
                double cost = 0.0;
                lhs_ptr v;

                for (v = head_lhs_node; v != tail_lhs_node; v++)
                    if (v->matched)
#ifdef    ROUND_COSTS
                        cost += v->matched->c_init;
#else
                        cost += v->matched->c;
#endif

                return (cost);
            }

            void store_results() {
                // count edges
                result_n = 0;
                for (lhs_ptr v = head_lhs_node; v != tail_lhs_node; v++) {
                    result_n++;
                }
                if (result_n == 0)
                    return;
                // initialize
                result_edge_index = new index_t[result_n];
                result_cost = 0;

                // save match result
                int i = 0;
                for (lhs_ptr v = head_lhs_node; v != tail_lhs_node; v++, i++) {
                    assert (i < result_n);
#ifdef    ROUND_COSTS
                    result_cost += (index_t) v->matched->c_init;
#else
                    result_cost += (index_t) v->matched->c;
#endif
                    result_edge_index[i] = v->matched->edge_index;
                }
            }

            int main() {
                //size_t time;


/*
(void) fprintf(stderr,"CSA: |>  n = %u,  m = %u,  sc_f = %lg", n, m, scale_factor);
*/
#if    defined(USE_PRICE_OUT) || defined(ROUND_COSTS)
                /*
                (void) fprintf(stderr,",  po_thr = %lg", po_cost_thresh);
                */
#endif
/*
(void) fprintf(stderr,"\n");
*/

#ifdef    PREC_COSTS
                min_epsilon = 2.0 / (double) (n + 1);
#else
                min_epsilon = 1.0 / (double) (n + 1);
#endif

                //time = myclock();

#ifdef    USE_P_REFINE
                (void) update_epsilon();
                refine();
#endif

                while (epsilon > min_epsilon) {
#ifdef    VERBOSE_TIME
                    /*
                      (void) fprintf(stderr,"CSA: |>   Epsilon = %lg; time = %lg\n",
                            epsilon, ((double) (myclock() - time)) / 60.0);
                    */
#endif
#ifdef    USE_P_REFINE
                    if (!update_epsilon() || !p_refine())
                      refine();
#else
                    (void) update_epsilon();
                    refine();
#endif
                }

                //time = myclock() - time;

                store_results();
                return (0);
            }

///////////////////////////////////////////////////////////////////////////
// debug.c ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
/*
    void show_lhs_node(int lhs_id) {
        lhs_ptr v = &head_lhs_node[lhs_id - 1];
        int rhs_id;
        rhs_ptr w;
        lr_aptr a;
        double v_price, this_price;

        (void) printf("Lhs node %d ", lhs_id);
        if (v->matched) {
            w = v->matched->head;
            rhs_id = w - head_rhs_node + 1;
            (void) printf("matched thru stored cost %lg to rhs node %d",
                          v->matched->c, rhs_id);
            if (w->matched == v)
                (void) putchar('\n');
            else {
                lhs_id = w->matched - head_lhs_node + 1;
                (void) printf(", matched back to lhs node %d\n", lhs_id);
            }
            (void) printf("\tMatching arc is priced ");
            if (v->matched->head->node_info.priced_in)
                (void) printf("in\n");
            else
                (void) printf("out\n");
        } else
            (void) printf("unmatched\n");
        (void) printf("\t%d arcs priced out, %d arcs priced in\n",
                      v->first - v->priced_out, (v + 1)->priced_out - v->first);
        if ((v + 1)->priced_out - v->first > 0) {
            (void) printf("\tPriced in arcs:\n");
            for (a = v->first; a == v->matched; a++);
            if (a == (v + 1)->priced_out)
                v_price = 0.0;
            else {
                v_price = a->head->p - a->c;
                for (a++; a != (v + 1)->priced_out; a++)
                    if ((a != v->matched) &&
                        (v_price < (this_price = a->head->p - a->c)))
                        v_price = this_price;
            }
            for (a = v->first; a != (v + 1)->priced_out; a++) {
                rhs_id = a->head - head_rhs_node + 1;
                (void) printf("\t\t(%d, %d) stored cost %lg, cmp cost %lg\n",
                              lhs_id, rhs_id, a->c, v_price + a->c - a->head->p);
            }
        }
    }

    void show_rhs_node(int rhs_id) {
        rhs_ptr v = &head_rhs_node[rhs_id - 1];
        int lhs_id;
#ifdef    USE_P_UPDATE
        rl_aptr	b;
#endif

        (void) printf("Rhs node %d, p %lg ",
                      rhs_id, v->p);
        if (v->matched) {
            lhs_id = v->matched - head_lhs_node + 1;
            if (v->matched->matched->head == v)
                (void) printf("matched thru stored cost %lg to lhs node %d\n",
                              v->matched->matched->c, lhs_id);
            else
                (void) printf("matched inconsistently to lhs node %d\n", lhs_id);
        } else
            (void) printf("unmatched\n");
#ifdef    USE_P_UPDATE
        for (b = v->priced_out; b != v->back_arcs; b++)
          {
          lhs_id = b->tail - head_lhs_node + 1;
          (void) printf("Arc (%ld, %ld) back stored cost %lg (priced out)\n",
                lhs_id, rhs_id, b->c);
          }
        for (; b != (v+1)->priced_out; b++)
          {
          lhs_id = b->tail - head_lhs_node + 1;
          (void) printf("Arc (%ld, %ld) back stored cost %lg (priced in) cmp cost %lg\n",
                lhs_id, rhs_id, b->c, b->c - v->p);
          }
#endif
    }

    void show_lhs() {
        int id;

        for (id = 1; id <= tail_lhs_node - head_lhs_node; id++)
            show_lhs_node(id);
    }

    void show_rhs() {
        int id;

        for (id = 1; id <= tail_rhs_node - head_rhs_node; id++)
            show_rhs_node(id);
    }

    int check_e_o_node(lhs_ptr v, double _epsilon) {
        lr_aptr a;
        double match_rc;
        int ret = FALSE;

        if (v->matched) {
            match_rc = v->matched->c - v->matched->head->p;
            for (a = v->first; a != (v + 1)->priced_out; a++) {
                if (a->c - a->head->p - match_rc < -epsilon * 1.01) {
                    (void) printf("Violated epsilon optimality: c(%d, %d)=%lg; matched to %d; eps=%lg\n",
                                  v - head_lhs_node + 1,
                                  a->head - head_rhs_node + 1,
                                  a->c - a->head->p - match_rc,
                                  v->matched->head - head_rhs_node + 1,
                                  _epsilon);
                    ret = TRUE;
                }
            }
        }
        return (ret);
    }

    void check_e_o(double _epsilon) {
        lhs_ptr v;

        for (v = head_lhs_node; v != tail_lhs_node; v++)
            (void) check_e_o_node(v, _epsilon);
    }
*/
///////////////////////////////////////////////////////////////////////////
// parse.c ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#define ERRBASE      1000    /* Base number for user-defined errors*/
#define BADINPUT1    1001    /* Bad input file format */
#define BADINPUT2    1002    /* Bad input file format */
#define BADINPUT3    1003    /* Bad input file format */
#define BADINPUT4    1004    /* Bad input file format */
#define BADINPUT5    1005    /* Bad input file format */
#define BADCOUNT     1006    /* Arc count discrepancy */
#define NONCONTIG    1007    /* Node id numbers not contiguous */
#define NOMEM        1008    /* Not enough memory */

            /*const std::string err_messages[8] = {
                    "Can't read from the input file.",
                    "Not a correct assignment problem line.",
                    "Error reading a node descriptor from the input.",
                    "Error reading an arc descriptor from the input.",
                    "Unknown line type in the input",
                    "Inconsistent number of arcs in the input.",
                    "Parsing noncontiguous node ID numbers not implemented.",
                    "Can't obtain enough memory to solve this problem.",
            };

            void parse_error(string s) {
                throw std::invalid_argument("CSA: Error while parsing the input: %s \n" + err_messages[(err_index % ERRBASE) - 1]);
                //(void) fprintf(stderr, "CSA: Error while parsing the input: %s \n",
                //               err_messages[(err_index % ERRBASE) - 1]);
                //exit(1);
            }*/

            typedef struct temp_arc {
                lhs_ptr tail;
                rhs_ptr head;
                index_t cost;
                index_t edge_index;
            } *ta_ptr;

            template<typename T1, typename T2>
            index_t parse(const xt::xexpression<T1> &xsources,
                          const xt::xexpression<T1> &xtargets,
                          const xt::xexpression<T2> &xweights) {
                auto &weights = xweights.derived_cast();
                auto &sources = xsources.derived_cast();
                auto &targets = xtargets.derived_cast();

                //size_t arc_count;
                size_t tail, head, lhs_n,
                        swap, id_offset, temp;
                int64_t cost, *lhs_degree;
#ifdef    STORE_REV_ARCS
                int64_t	*rhs_degree;
                rl_aptr	b;
#endif
                index_t max_cost = 0;
                lr_aptr a;
                ta_ptr temp_a, temp_arcs;
                lhs_ptr l_v;
                rhs_ptr r_v;

                //arc_count = m;
                lhs_n = n / 2;
                if (lhs_n == 0)
                    return 0;

                head_lr_arc = (lr_aptr) malloc((m + 1) * sizeof(struct lr_arc));
                tail_lr_arc = head_lr_arc + m;
#ifdef    STORE_REV_ARCS
                head_rl_arc = (rl_aptr) malloc((m + 1) * sizeof(struct rl_arc));
                tail_rl_arc = head_rl_arc + m;
#endif
                id_offset = lhs_n;
                if (lhs_n > n - lhs_n) {
                    lhs_n = n - lhs_n;
                    swap = TRUE;
                } else
                    swap = FALSE;
                head_lhs_node = (lhs_ptr) malloc((lhs_n + 1) *
                                                 sizeof(struct lhs_node));
                tail_lhs_node = head_lhs_node + lhs_n;
                head_rhs_node = (rhs_ptr) malloc((n - lhs_n + 1) *
                                                 sizeof(struct rhs_node));
                tail_rhs_node = head_rhs_node + n - lhs_n;
                lhs_degree = (int64_t *) malloc(lhs_n * sizeof(int64_t));
#ifdef    STORE_REV_ARCS
                rhs_degree = (int64_t *) malloc((n - lhs_n) * sizeof(int64_t));
                if ((rhs_degree == NULL) || (head_rl_arc == NULL))
                  parse_error(NOMEM);
                for (tail = 0; tail < n - lhs_n; tail++)
                  rhs_degree[tail] = 0;
#endif
                temp_arcs = (ta_ptr) malloc(m * sizeof(struct temp_arc));
                if ((head_lhs_node == NULL) || (head_lr_arc == NULL) ||
                    (lhs_degree == NULL) || (temp_arcs == NULL))
                    throw std::invalid_argument("CSA: Error while parsing the input: memory allocation failed.");
                temp_a = temp_arcs;
                for (tail = 0; tail < lhs_n; tail++)
                    lhs_degree[tail] = 0;


                for (size_t i = 0; i < m; i++) {

                    tail = sources(i) + 1;
                    head = targets(i) + 1;
                    cost = (int64_t) weights(i);

                    head -= id_offset;
                    if (swap) {
                        temp = head;
                        head = tail;
                        tail = temp;
                    }
                    if ((tail < 1) || (tail > lhs_n) ||
                        (head < 1) || (head > n - lhs_n))
                        throw std::invalid_argument("CSA: Error while parsing the input: Invalid edge descriptors.");
                        

                    head--;
                    tail--;
                    temp_a->head = head_rhs_node + head;
                    temp_a->tail = head_lhs_node + tail;
                    temp_a->cost = cost;
                    temp_a->edge_index = i;
                    if ((cost = abs((int) cost)) > max_cost) max_cost = cost;
                    temp_a++;
                    lhs_degree[tail]++;
#ifdef    STORE_REV_ARCS
                    rhs_degree[head]++;
#endif
                }


                a = head_lr_arc;
                for (tail = 0, l_v = head_lhs_node; l_v != tail_lhs_node; l_v++, tail++) {
                    l_v->priced_out = l_v->first = a;
                    l_v->matched = nullptr;
                    a += lhs_degree[tail];
#ifdef    QUICK_MIN
                    if (lhs_degree[tail] < NUM_BEST + 1)
                        l_v->node_info.few_arcs = TRUE;
                    else
                        l_v->node_info.few_arcs = FALSE;
#endif
                }
                tail_lhs_node->priced_out = a;

#ifdef    STORE_REV_ARCS
                tail = 0;
                b = head_rl_arc;
#endif
                for (r_v = head_rhs_node; r_v != tail_rhs_node; r_v++) {
                    r_v->node_info.priced_in = TRUE;
                    r_v->matched = NULL;
#ifdef    STORE_REV_ARCS
                    r_v->priced_out = r_v->back_arcs = b;
                    b += rhs_degree[tail];
                    tail++;
#endif
#ifdef    ROUND_COSTS
                    r_v->base_p = 0.0;
                    r_v->p = 0;
#else
                    r_v->p = 0.0;
#endif
                }
#ifdef    STORE_REV_ARCS
                tail_rhs_node->priced_out = b;
#endif

                for (temp_a--; temp_a != temp_arcs - 1; temp_a--) {
                    a = temp_a->tail->first + (--lhs_degree[temp_a->tail - head_lhs_node]);
                    a->head = temp_a->head;
#ifdef    ROUND_COSTS
#ifdef	MIN_COST
                    a->c_init = (double) temp_a->cost;
#else
                    a->c_init = (double) -temp_a->cost;
#endif
#else	/* PREC_COSTS */
#ifdef    MIN_COST
                    a->c = (double) temp_a->cost;
                    a->edge_index = temp_a->edge_index;
#else
                    a->c = (double) -temp_a->cost;
#endif
#endif    /* ROUND_COSTS */
#ifdef    USE_SP_AUG_FORWARD
                    a->tail = temp_a->tail;
#endif
#ifdef    STORE_REV_ARCS
                    b = temp_a->head->back_arcs + (--rhs_degree[temp_a->head - head_rhs_node]);
                    a->rev = b;
#if	defined(ROUND_COSTS) || defined(USE_PRICE_OUT) || \
    defined(USE_SP_AUG_BACKWARD)
                    b->rev = a;
#endif
                    b->tail = temp_a->tail;
#if	defined(USE_P_UPDATE) || defined(USE_SP_AUG_BACKWARD)
                  /*
                  In the ROUND_COSTS case, update_epsilon() takes care of b->c.
                  */
#ifdef	PREC_COSTS
                    b->c = a->c;
#endif
#endif
#endif
                }

                (void) free((char *) temp_arcs);
                (void) free((char *) lhs_degree);
#ifdef    STORE_REV_ARCS
                (void) free((char *) rhs_degree);
#endif

                return (max_cost);
            }

///////////////////////////////////////////////////////////////////////////
// refine.c ///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#ifdef    QUICK_MIN
#define    sort_insert(best, size, a, a_prc, nsize) \
\
{\
size_t    si_i, si_j;\
\
if (size == 0)\
  best[0] = a;\
else\
  {\
  si_j = size;\
  for (si_i = 0; si_i < size; si_i++)\
    if (a_prc < best[si_i]->c - best[si_i]->head->p)\
      {\
      si_j = si_i;\
      for (si_i = nsize - 1; si_i > si_j; si_i--)\
    best[si_i] = best[si_i - 1];\
      break;\
      }\
  best[si_j] = a;\
  }\
}
#endif

#ifdef    QUICK_MIN

            void best_build(lhs_ptr v) {
                size_t i;
                lr_aptr a, a_stop;
                double red_cost, save_max;

                rebuilds++;
                for (i = 0, a = v->first; i < NUM_BEST; i++, a++) {
                    red_cost = a->c - a->head->p;
                    sort_insert(v->best, i, a, red_cost, i + 1);
                }
#ifdef    LOOSE_BOUND
                v->next_best = v->best[NUM_BEST - 1]->c -
                           v->best[NUM_BEST - 1]->head->p;
#else
/*
Calculate initial next_best by looking at the next arc in the
adjacency list.
*/
                if ((v->next_best = a->c - a->head->p) <
                    (red_cost = v->best[NUM_BEST - 1]->c -
                                v->best[NUM_BEST - 1]->head->p)) {
                    sort_insert(v->best, NUM_BEST, a, v->next_best, NUM_BEST);
                    v->next_best = red_cost;
                }
                a++;
#endif
/*
Now go through remaining arcs in adjacency list and place each one
at the appropriate place in best[], if any.
*/
                a_stop = (v + 1)->priced_out;
                for (; a != a_stop; a++) {
                    if ((red_cost = a->c - a->head->p) < v->next_best)
#ifdef    LOOSE_BOUND
                        {
                        sort_insert(v->best, NUM_BEST, a, red_cost, NUM_BEST);
                        v->next_best = v->best[NUM_BEST - 1]->c -
                               v->best[NUM_BEST - 1]->head->p;
                        }
#else
                    {
                        if (red_cost < (save_max = v->best[NUM_BEST - 1]->c -
                                                   v->best[NUM_BEST - 1]->head->p)) {
                            sort_insert(v->best, NUM_BEST, a, red_cost, NUM_BEST);
                            v->next_best = save_max;
                        } else
                            v->next_best = red_cost;
                    }
#endif
                }
            }

#endif

/* Assume v has excess (is unassigned) and do a double push from v. */

            void double_push(lhs_ptr v) {
                double v_pref, v_second, red_cost, adm_gap;
                lr_aptr a, a_stop, adm;
                rhs_ptr w;
                lhs_ptr u;
#ifdef    QUICK_MIN
                size_t i;
                lr_aptr *check_arc;
#endif

#ifdef    DEBUG
                (void) printf("%lu p's, %lu dp's: dp on %ld ", pushes, double_pushes,
                          v - head_lhs_node + 1);
#endif

/*
Begin part I: Compute the following:
  o adm, the minimum-reduced-cost arc incident to v,
  o adm_gap, the amount by which the reduced cost of adm must be
    increased to make it equal in reduced cost to another arc incident
    to v, or enough to price the arc out if it is the only incident
    arc.
*/

#ifdef    QUICK_MIN
                if (v->node_info.few_arcs) {
                    scans++;
#endif

                    /*
                    If the input problem is feasible, it is never the case that
                    (a_stop == a) after the following two lines because we never get
                    excess at a node with no incident arcs.
                    */
                    a_stop = (v + 1)->priced_out;
                    a = v->first;
                    v_pref = a->c - a->head->p;
                    v_second = v_pref + epsilon * (po_cost_thresh + 1.0);
                    adm = a;
                    /*
                    After this loop, v_pref is the minimum reduced cost of an edge out of
                    v, and v_second is the second-to-minimum such reduced cost.
                    */
                    for (a++; a != a_stop; a++)
                        if (v_pref > (red_cost = a->c - a->head->p)) {
                            v_second = v_pref;
                            v_pref = red_cost;
                            adm = a;
                        } else if (v_second > red_cost)
                            v_second = red_cost;

#ifdef    QUICK_MIN
                } else {
                    /*
                    Find the minimum and second-minimum edges listed in the node's
                    best[] array, and check whether their present partial reduced
                    costs are below the node's bound as stored in next_best. If they
                    are, we calculate adm_gap and are done with part I. If not, we
                    rebuild the best[] array and the next_best bound, and calculate the
                    required information.
                    */
                    adm = v->best[1];
                    v_second = adm->c - adm->head->p;
                    adm = v->best[0];
                    v_pref = adm->c - adm->head->p;
                    if (v_pref > v_second) {
                        adm = v->best[1];
                        red_cost = v_second;
                        v_second = v_pref;
                        v_pref = red_cost;
                    }
#if    (NUM_BEST > 2)
                    for (i = NUM_BEST - 2, check_arc = &v->best[2]; i > 0; i--, check_arc++) {
                        a = *check_arc;
                        if (v_pref > (red_cost = a->c - a->head->p)) {
                            v_second = v_pref;
                            v_pref = red_cost;
                            adm = a;
                        } else if (v_second > red_cost)
                            v_second = red_cost;
                    }
#endif
                    if (v_second > v->next_best) {
                        /*
                        Rebuild the best[] array and recalculate next_best.
                        !v->node_info.few_arcs, so we know there are enough incident arcs
                        to fill up best[] initially and have one left over for next_best.
                        */
                        best_build(v);
                        adm = v->best[1];
                        v_second = adm->c - adm->head->p;
                        adm = v->best[0];
                        v_pref = adm->c - adm->head->p;
                    } else
                        non_scans++;
                }
#endif

                adm_gap = v_second - v_pref;

#ifdef    EXPLICIT_LHS_PRICES
                if (v->p + v_pref > epsilon)
                  {
                  /*
                  v needs relabeling, so we rack up a relabeling that we wouldn't if
                  we were using implicit prices. This relabeling makes adm an
                  admissible arc.
                  */
                  unnec_rel++;
                  v->p = -v_pref;
                  }
#endif

#ifdef    DEBUG
                (void) printf("to %lu. Gap = %lg\n", adm->head - head_rhs_node + 1, adm_gap);
#endif

/*
Begin part II: Using the information computed in part I,
  o match v to w, adm's head, and
  o unmatch the node (if any) formerly matched to w.
In the case where w's current matching arc is priced out, we do not
change the matching, but we reset the value of adm_gap so that the
(v, w) arc will be priced out.
*/
                w = adm->head;
                if ((u = w->matched))
                    /*
                    If w's matched arc is priced in, go ahead and unmatch (u, w) and
                    match (v, w). If w's matched arc is priced out, abort the double
                    push and relabel w so v no longer prefers w.
                    */
                    if (w->node_info.priced_in) {
                        pushes += 2;
                        double_pushes++;
                        u->matched = NULL;
                        make_active(u);
                        v->matched = adm;
                        w->matched = v;
                    } else {
                        adm_gap = epsilon * po_cost_thresh;
                        make_active(v);
                    }
                else {
                    total_e--;
                    pushes++;
                    v->matched = adm;
                    w->matched = v;
                }

#ifdef    EXPLICIT_LHS_PRICES
                /*
                Relabel v: v's price is chosen to make the reduced cost of v's new
                preferred arc (v->p + v_pref + adm_gap) equal to zero.
                */
                relabelings++;
                v->p = -(v_pref + adm_gap);
#endif

/*
Relabel w: v's price is chosen to make the implicit reduced cost of
v's new preferred arc (v_pref + adm_gap) equal to zero. Then w's price
is chosen so that the arc just matched has implicit reduced cost
-epsilon.
*/
                relabelings++;
                w->p -= adm_gap + epsilon;
            }

            void refine() {
                lhs_ptr v;
#ifdef    USE_P_UPDATE
                WORK_TYPE	old_refine_work_upd;
#endif
#ifdef    STRONG_PO
                WORK_TYPE	old_refine_work_po;
#endif

                //refine_time -= myclock();
                refines++;
/*
Saturate all negative arcs: Negative arcs are exactly those
right-to-left matching arcs with negative reduced cost, and there is
an interpretation of the implicit price function on the left that
admits all right-to-left matching arcs. This interpretation is
always consistent with the stored prices of lhs nodes in the case
of EXPLICIT_PRICES.
*/
                total_e = 0;
                for (v = head_lhs_node; v != tail_lhs_node; v++) {
                    if (v->matched && v->matched->head->node_info.priced_in) {
                        v->matched->head->matched = NULL;
                        v->matched = NULL;
                    }
                    if (v->matched == NULL) {
                        total_e++;
                        make_active(v);
                    }
                }

#ifdef    USE_P_UPDATE
                old_refine_work_upd = REFINE_WORK;
#endif
#ifdef    STRONG_PO
                old_refine_work_po = REFINE_WORK;
#endif

#ifdef    STRONG_PO
                while ((total_e > 0) || (old_refine_work_po = REFINE_WORK,
                             !check_po_arcs()))
#else
                while (total_e > EXCESS_THRESH)
#endif
                {
#ifdef    USE_P_UPDATE
                    if (REFINE_WORK - old_refine_work_upd > upd_work_thresh)
                      {
                      old_refine_work_upd = REFINE_WORK;
                      p_update();
#if	defined(DEBUG) && defined(CHECK_EPS_OPT)
                      check_e_o(epsilon);
#endif
                      }
#endif
#ifdef    STRONG_PO
                    if (REFINE_WORK - old_refine_work_po > po_work_thresh)
                      {
                      old_refine_work_po = REFINE_WORK;
                      (void) check_po_arcs();
#if	defined(DEBUG) && defined(CHECK_EPS_OPT)
                      check_e_o(epsilon);
#endif
                      }
#endif
                    get_active_node(v);
                    double_push(v);
                }

#ifdef    USE_SP_AUG
                if (total_e > 0)
                  sp_aug();
#endif

                //refine_time += myclock();
            }

///////////////////////////////////////////////////////////////////////////
// stack.c ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

            //const char *nomem_msg = "Insufficient memory.\n";

            void st_reset(stack s) {
                s->top = s->bottom;
            }

            char *st_pop(stack s) {
                s->top--;
                return (*(s->top));
            }

            stack st_create(size_t size) {
                stack s;

                s = (stack) malloc(sizeof(struct stack_st));

                if (s == NULL) {
                    throw std::runtime_error("Insufficient memory.");
                    //(void) fprintf(stderr, "%s", nomem_msg);
                    //exit(9);
                }
                s->bottom = (char **) malloc(size * sizeof(char *));
                if (s->bottom == NULL) {
                    free(s);
                    throw std::runtime_error("Insufficient memory.");
                    //(void) fprintf(stderr, "%s", nomem_msg);
                    // exit(9);
                }
                s->top = s->bottom;
                return (s);
            }

            void st_destroy(stack s) {
                free(s->bottom);
                free(s);
            }

///////////////////////////////////////////////////////////////////////////
// timer.c ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
/*
            size_t myclock() {
                struct tms hold;

                (void) times(&hold);
                return (hold.tms_utime);
            }
*/
///////////////////////////////////////////////////////////////////////////
// update_epsilon.c ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

            int update_epsilon() {
#ifdef    USE_PRICE_OUT
                double	po_cutoff;
                double	thresh;
                int	one_priced_in;
#if	defined(QUICK_MIN) && !defined(BACK_PRICE_OUT)
                int	need_best_rebuild;
#endif	/* QUICK_MIN && !BACK_PRICE_OUT */
#ifdef	BACK_PRICE_OUT
                rl_aptr	b, b_stop;
                lhs_ptr	u;
                rhs_ptr	w;
#endif	/* BACK_PRICE_OUT */
#endif    /* USE_PRICE_OUT */
#if    defined(USE_PRICE_OUT) || defined(CHECK_EPS_OPT)
                double	v_price, red_cost;
                lhs_ptr	v;
                lr_aptr	a, a_start, a_stop;
#endif    /* USE_PRICE_OUT || CHECK_EPS_OPT */
#ifdef    STRONG_PO
                double	fix_in_thresh;
#endif    /* STRONG_PO */

                int fix_in = FALSE;

                epsilon /= scale_factor;

                if (epsilon < min_epsilon) epsilon = min_epsilon;

#ifdef    USE_PRICE_OUT
                po_cutoff = po_cost_thresh * epsilon;
#endif

#if    defined(USE_PRICE_OUT) || defined(CHECK_EPS_OPT)
                /*
                Now if at least one refine has occurred (and hence all lhs nodes are
                matched), check for arcs that should be priced in and price them in,
                and check for arcs that should be priced out and price them out.
                */
                if (refines > 0)
                  {
                  for (v = head_lhs_node; v != tail_lhs_node; v++)
                    {
                    /*
                    First, save the location of the first priced-in arc so we don't do
                    costly computations for any arc twice.
                    */
                    a_start = v->first;
                    /*
                    Determine the price we will assume v has. We choose the price so the
                    matching arc will have zero partial reduced cost, and this choice
                    enables us to do several things:
                    1) Detect epsilon-optimality by checking that partial reduced costs
                       of non-matching arcs are at least -epsilon;
                    2) Make price-in and price-out decisions in a way that lets us store
                       new costs (with incorporated rhs prices) without any backtracking.
                       Those arcs with reduced cost close to that of the matching arc
                       are priced in, those far away are priced out, and that's that.
                    */
                    v_price = v->matched->head->p - v->matched->c;
#ifdef	USE_PRICE_OUT
                    thresh = po_cutoff - v_price;
                    one_priced_in = FALSE;
#if	defined(QUICK_MIN) && !defined(BACK_PRICE_OUT)
                    need_best_rebuild = FALSE;
#endif	/* QUICK_MIN && !BACK_PRICE_OUT */
#endif	/* USE_PRICE_OUT */
#ifdef	STRONG_PO
                    /*
                    Check for arcs to price in.
                    */
                    fix_in_thresh = -epsilon - v_price;
                    for (a = v->priced_out; a != v->first; a++)
                      if ((a != v->matched) && ((red_cost = a->c - a->head->p) < thresh))
                    {
                    price_in_unm_arc(v, a);
                    one_priced_in = TRUE;
                    /*
                    If we have a fix-in, we don't have to unmatch the node here,
                    since refine unmatches all nodes initially anyway. Just let
                    main know that refine is needed.
                    */
                    if (red_cost < fix_in_thresh)
                      fix_in = TRUE;
#if	defined(QUICK_MIN) && !defined(BACK_PRICE_OUT)
                    need_best_rebuild = TRUE;
#endif	/* QUICK_MIN && !BACK_PRICE_OUT */
                    if (a == v->first) break;
                    }
#endif	/* STRONG_PO */
                    a_stop = (v+1)->priced_out;
                    /*
                    For each arc incident to v, decide whether or not to price it out.
                    */
                    for (a = a_start; a != a_stop; a++)
                      {
                      if (a != v->matched)
                    {
                    red_cost = a->c - a->head->p;
#ifdef	USE_PRICE_OUT
                    if (red_cost >= thresh)
                      {
                      price_out_unm_arc(v, a);
#if	defined(QUICK_MIN) && !defined(BACK_PRICE_OUT)
                      need_best_rebuild = TRUE;
#endif	/* QUICK_MIN && !BACK_PRICE_OUT */
                      }
                    else
                      {
                      one_priced_in = TRUE;
#endif	/* USE_PRICE_OUT */
#ifdef	CHECK_EPS_OPT
                      /*
                      0.01 in the following line because of precision problems
                      that are ultimately OK, although they make the flow look
                      non-epsilon-optimal.
                      */
                      if (v_price + red_cost < -epsilon * (scale_factor + 0.01))
                        {
                        (void) printf("Epsilon optimality violation! c(%ld, %ld)=%lg; epsilon=%lg\n",
                              v - head_lhs_node + 1, a->head - head_rhs_node + 1,
                              v_price + red_cost, epsilon);
                        (void) fflush(stdout);
                        }
#endif	/* CHECK_EPS_OPT */
#ifdef	USE_PRICE_OUT
                      }
#endif	/* USE_PRICE_OUT */
                    }
                      }
#ifdef	USE_PRICE_OUT
                    a = v->matched;
#ifdef	STRONG_PO
                    if (one_priced_in)
                      {
                      if (!a->head->node_info.priced_in)
                    {
                    /*
                    Matching arc is priced out.
                    */
                    price_in_mch_arc(v, a);
#if	defined(QUICK_MIN) && !defined(BACK_PRICE_OUT)
                    need_best_rebuild = TRUE;
#endif	/* QUICK_MIN && !BACK_PRICE_OUT */
                    }
                      }
                    else
#else	/* !STRONG_PO */
                    if (!one_priced_in)
#endif	/* STRONG_PO */
                      if (a->head->node_info.priced_in)
                    {
                    /*
                    No arcs are priced in except the matching arc. Price it out,
                    too, and if we use back-arc price-outs, price out all the arcs
                    incident to its head.
                    */
                    price_out_mch_arc(v, a);
#if	defined(QUICK_MIN) && !defined(BACK_PRICE_OUT)
                    need_best_rebuild = TRUE;
#endif	/* QUICK_MIN && !BACK_PRICE_OUT */
#ifdef	BACK_PRICE_OUT
                    w = a->head;
                    b_stop = (w+1)->priced_out;
                    for (b = w->back_arcs; b != b_stop; b++)
                      {
                      u = b->tail;
                      a = b->rev;
                      price_out_unm_arc(u, a);
                      }
#endif	/* BACK_PRICE_OUT */
                    }
#if	defined(QUICK_MIN) && !defined(BACK_PRICE_OUT)
                    /*
                    Make sure v->node_info.few_arcs reflects the priced-in degree of v.
                    */
                    if (a_stop - v->first < NUM_BEST + 1)
                      v->node_info.few_arcs = TRUE;
                    else
                      {
                      v->node_info.few_arcs = FALSE;
                      if (need_best_rebuild)
                    best_build(v);
                      }
#endif	/* QUICK_MIN && !BACK_PRICE_OUT */
#endif	/* USE_PRICE_OUT */
                    }
#if	defined(QUICK_MIN) && defined(BACK_PRICE_OUT)
                  /*
                  Rebuild the best list for every node, since back price outs mean we
                  can't build them as we do the price outs.
                  */
                  for (v = head_lhs_node; v != tail_lhs_node; v++)
                    if ((v+1)->priced_out - v->first < NUM_BEST + 1)
                      v->node_info.few_arcs = TRUE;
                    else
                      {
                      v->node_info.few_arcs = FALSE;
                      best_build(v);
                      }
#endif
                  }
#endif    /* USE_PRICE_OUT || CHECK_EPS_OPT */
                return (!fix_in);
            }

///////////////////////////////////////////////////////////////////////////
// check_po_arcs.c ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#ifdef    STRONG_PO

            int	check_po_arcs()

            {
            lhs_ptr	v;
            lr_aptr	a, a_start, a_stop;
            int	one_priced_in, fix_this_node, fix_in = FALSE;
            double	match_rc, this_cost, v_price, this_price, po_cutoff, thresh,
                fix_in_thresh;
#ifdef	QUICK_MIN
            int	need_best_rebuild;
#endif

#ifdef	DEBUG
            (void) printf("Checking priced-out arcs. total_e=%lu\n", total_e);
#endif

            po_cutoff = po_cost_thresh * epsilon;
            for (v = head_lhs_node; v != tail_lhs_node; v++)
              {
#ifdef	QUICK_MIN
              need_best_rebuild = FALSE;
#endif
              /*
              All routines that incorporate prices into stored costs must update
              stored costs of priced-out arcs so the following code correctly
              computes reduced costs of priced-out arcs. At the present time,
              there are no such routines.
              */
              a_stop = (v+1)->priced_out;
              if (a = v->matched)
                {
                /*
                Node v is matched. Price in any arcs not far costlier than the
                matching arc, and if there are any such arcs, make sure the
                matching arc is priced in, too.
                */
                a_start = v->first;
                one_priced_in = (a_start != a_stop);
                fix_this_node = FALSE;
                match_rc = a->c - a->head->p;
                thresh = match_rc + po_cutoff;
                fix_in_thresh = match_rc - epsilon;
                for (a = v->priced_out; a != v->first; a++)
                  if ((a != v->matched) && ((this_cost = a->c - a->head->p) < thresh))
                {
                price_in_unm_arc(v, a);
                one_priced_in = TRUE;
                if (this_cost < fix_in_thresh)
                  {
                  /*
                  Epsilon-optimality violated by priced-in arc.
                  */
                  fix_in = TRUE;
                  fix_this_node = TRUE;
#ifdef	DEBUG
                  (void) printf("Fixing in arc (%ld, %ld)\n", v - head_lhs_node + 1,
                        a->head - head_rhs_node + 1);
#endif
                  }
#ifdef	QUICK_MIN
                need_best_rebuild = TRUE;
#endif
                /*
                If we priced in the last priced-out arc in the list,
                a == v->first, and we need to keep a from advancing too far.
                */
                if (a == v->first) break;
                }
                /*
                Now if matching arc is priced out and there is some arc now priced
                in that has a reduced cost not far enough above that of the
                matching arc, price in the matching arc. We already know this
                condition on arcs we priced in, of course. Don't check them.
                */
                if (!v->matched->head->node_info.priced_in)
                  if (one_priced_in)
                {
                a = v->matched;
                price_in_mch_arc(v, a);
#ifdef	QUICK_MIN
                need_best_rebuild = TRUE;
#endif
                }
                /*
                If a fix-in occurred on this node, unmatch it to preserve
                epsilon-optimality.
                */
                if (fix_this_node)
                  {
#ifdef	DEBUG
                  (void) printf("Fix-in -- unmatching (%ld, %ld)\n",
                        v - head_lhs_node + 1,
                        v->matched->head - head_rhs_node + 1);
#endif
                  v->matched->head->matched = NULL;
                  v->matched = NULL;
                  total_e++;
                  make_active(v);
                  }
                }
              else
                {
                /*
                Node v is unmatched. Price any arc in whose reduced cost is less
                than po_cutoff above the minimum priced-in arc.
                */
                a = v->first;
                if (a != a_stop)
                  {
#ifdef	EXPLICIT_LHS_PRICES
                  v_price = v->p;
#else
                  v_price = a->head->p - a->c;
                  for (a++; a != a_stop; a++)
                if (v_price < (this_price = a->head->p - a->c))
                  v_price = this_price;
#endif
                  for (a = v->priced_out; a != v->first; a++)
                if (v_price - (this_price = a->head->p - a->c) < po_cutoff)
                  {
                  price_in_unm_arc(v, a);
                  /*
                  If (this_price > v_price), we might have priced in some arcs
                  unnecessarily because the reduced costs of arcs incident to
                  v turn out to be higher than we thought. This is OK, but do
                  the right thing for the rest of the priced-out arcs.
                  */
                  if (this_price > v_price)
                    v_price = this_price;
#ifdef	QUICK_MIN
                  need_best_rebuild = TRUE;
#endif
                  /*
                  If we priced in the last priced-out arc in the list,
                  a == v->first, and we need to keep a from advancing too far.
                  */
                  if (a == v->first) break;
                  }
                  }
                }
#ifdef	QUICK_MIN
              /*
              Make sure v->node_info.few_arcs reflects the priced-in degree of v.
              */
              if (a_stop - v->first < NUM_BEST + 1)
                v->node_info.few_arcs = TRUE;
              else
                {
                v->node_info.few_arcs = FALSE;
                if (need_best_rebuild)
                  best_build(v);
                }
#endif
              }

#ifdef	DEBUG
            (void) printf("Checked priced-out arcs. total_e=%lu\n", total_e);
#endif

            if (fix_in) fix_ins++;
            return(!fix_in);
            }

#endif

///////////////////////////////////////////////////////////////////////////
// p_update.c /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#if defined(USE_P_UPDATE)

            /*
            Doing a u_scan on w updates the current estimate of required price
            changes on nodes adjacent (in the rhs sense) to w to establish deficit
            reachability in the admissible graph for all excesses.
            */

            size_t	u_scan(rhs_ptr w)

            {
            register	rl_aptr	b, b_stop;
            register	lr_aptr	a;
            register	rhs_ptr	u;
            lhs_ptr		v;
            register	int64_t	wk, uk;
            register	double	p;
            double	u_to_w_cost;
            size_t	excess_found = 0;

            u_scans++;
            b_stop = (w+1)->priced_out;
            p = w->p;
            wk = w->key;
            for (b = w->back_arcs; b != b_stop; b++)
              if (a = b->tail->matched)
                {
                if (((u = a->head) != w) && u->node_info.priced_in)
                  {
#ifdef	P_U_ZERO_BACK_MCH_ARCS
                  u_to_w_cost = u->p + b->c - p;
#else
                  u_to_w_cost = u->p - a->rev->c + b->c - p;
#endif
                  if (u->key >= 0)
                {
                if (u_to_w_cost < 0.0)
                  uk = wk;
                else
                  /*
                  Preliminary check to make sure we're in the ballpark to
                  avoid overflow and to avoid costly double-to-long casts if
                  we don't need them.
                  */
                  if (epsilon * (u->key - wk) > u_to_w_cost)
                    uk = wk + 1 + (int64_t) (u_to_w_cost / epsilon);
                  else
                    uk = u->key;
                if (u->key > uk)
                  {
                  if (u->key != num_buckets)
                    delete_list(u, &bucket[u->key]);
                  u->key = uk;
                  insert_list(u, &bucket[uk]);
                  }
                }
                  }
                }
              else
                /*
                Encountered an excess -- b's tail isn't matched. Determine what
                price decrease on b's tail would be required to make the edge to w
                admissible. Recall that back arc costs are offset so preferred
                arcs have zero partial reduced cost at this point, so we need only
                examine the stored cost of the present edge, rather than compute
                the minimum. Avoid costly ceiling and cast when possible; also
                avoid overflows.
                */
                if ((u_to_w_cost = b->c - p) < epsilon * ((v = b->tail)->delta_reqd - wk))
                  {
                  uk = wk + (int64_t) ceil(u_to_w_cost / epsilon);
                  if (uk < v->delta_reqd)
                {
                if (uk == 0)
                  {
                  excess_found++;
#ifdef	DEBUG
                  (void) printf("claiming excess at node %ld\n",
                        v - head_lhs_node + 1);
#endif
                  }
                v->delta_reqd = uk;
                }
                  }

            w->p -= epsilon * w->key;
            w->key = -1;
            return(excess_found);
            }

            void	p_update()

            {
            rhs_ptr	w;
            lhs_ptr	v;
            double	delta_c, this_cost;
            int64_t	balance, level;
            lr_aptr	a, a_stop;

            p_update_time -= myclock();
            p_updates++;

#ifdef	DEBUG
            (void) printf("Doing p_update(): epsilon = %lg, total_e = %lu\n",
                      epsilon, total_e);
#endif

            for (v = head_lhs_node; v != tail_lhs_node; v++)
              {
              a_stop = (v+1)->priced_out;
#ifdef	P_U_ZERO_BACK_MCH_ARCS
              if (v->matched)
                {
                if (v->matched->head->node_info.priced_in)
                  {
                  /*
                  Offset back arc costs so back matching arc has zero stored cost
                  */
                  delta_c = v->matched->rev->c;
#ifdef	STRONG_PO
                  /*
                  In the case of strong price-outs, we could price in a back arc
                  later that's priced out now. So offset the costs of all the
                  incident back arcs.
                  */
                  a = v->priced_out;
#else
                  a = v->first;
#endif
                  for (; a != a_stop; a++)
                a->rev->c -= delta_c;
                  }
                }
              else
#else
              if (!v->matched)
#endif
                {
#ifdef	DEBUG
                (void) printf("excess at node %ld\n", v - head_lhs_node + 1);
#endif
                v->delta_reqd = num_buckets;
                a = v->first;
#ifdef	NO_FEAS_PROMISE
                if (a == a_stop)
                  {
                  (void) printf("Infeasible problem\n");
                  exit(9);
                  }
#endif
                delta_c = a->rev->c - a->head->p;
                for (a++; a != a_stop; a++)
                  if ((this_cost = a->rev->c - a->head->p) < delta_c)
                delta_c = this_cost;
#ifdef	STRONG_PO
                a_stop = v->priced_out - 1;
#else
                a_stop = v->first - 1;
#endif
                for (a--; a != a_stop; a--)
                  a->rev->c -= delta_c;
                }
              }

            for (w = head_rhs_node; w != tail_rhs_node; w++)
              if (w->matched)
                w->key = num_buckets;
              else
                {
                w->key = 0;
                insert_list(w, &bucket[0]);
                }

            balance = -total_e;
            level = 0;

            while ((balance < 0) && (level < num_buckets))
              if (bucket[level] == tail_rhs_node)
                level++;
              else
                {
                w = deq_list(&bucket[level]);
                balance += u_scan(w);
                }

            /*
            Now figure out by how much we need to decrease prices of nodes that
            didn't get scanned.
            */
            for (v = head_lhs_node; v != tail_lhs_node; v++)
              {
              if (!v->matched)
                {
                if (v->delta_reqd == num_buckets)
                  (void) printf("%u : excess at node %ld unclaimed after scans!\n",
                        p_updates, v - head_lhs_node + 1);
                if (v->delta_reqd > level)
                  level = v->delta_reqd;
                }
              }

            delta_c = level * epsilon;
            for (w = head_rhs_node; w != tail_rhs_node; w++)
              {
              if ((w->key != num_buckets) && (w->key >= 0))
                delete_list(w, &bucket[w->key]);
              if (w->key >= 0)
                w->p -= delta_c;
              }

            p_update_time += myclock();
            }

#endif

///////////////////////////////////////////////////////////////////////////
// sp_aug_backward.c //////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#if defined(USE_SP_AUG_BACKWARD)

            lhs_ptr	closest_node;
            index_t	int64_t	closest_dist;
            rhs_ptr	scanned;
            size_t	int64_t	level;	/* level currently being scanned */

            /*
            augment() moves a unit of excess from v along an augmenting path to
            cancel a deficit.
            */

            void	augment(lhs_ptr v)

            {
            lhs_ptr	x;
            rhs_ptr	w;

#ifdef	DEBUG
            (void) printf("augment(%ld):\n", v - head_lhs_node + 1);
#endif

            do
              {
              w = v->aug_path->head;
#if	defined(DEBUG) || defined(LOG_PATHS)
              (void) printf("%ld %ld, ", v - head_lhs_node + 1,
                    w - head_rhs_node + 1);
#endif
              v->matched = v->aug_path;
              x = w->matched;
              w->matched = v;
              v = x;
              }
            while (v);
#if	defined(DEBUG) || defined(LOG_PATHS)
            putchar('\n');
#endif
            }

            /*
            Doing an a_scan on w updates the current estimate of required price
            changes on nodes adjacent (in the rhs sense) to w to establish deficit
            reachability in the admissible graph for all excesses.
            */

            void	a_scan(rhs_ptr w)

            {
            register	rl_aptr	b, b_stop;
            register	lr_aptr	a;
            register	rhs_ptr	u;
            lhs_ptr		v;
            register	int64_t	wk, uk;
            register	double	p;
            double	u_to_w_cost;

#ifdef	DEBUG
            (void) printf("doing a_scan(%ld) key=%ld\n", w - head_rhs_node + 1, w->key);
#endif

            a_scans++;
            b_stop = (w+1)->priced_out;
            p = w->p;
            wk = w->key;
            for (b = w->back_arcs; b != b_stop; b++)
              if (a = b->tail->matched)
                {
                if (((u = a->head) != w) && u->node_info.priced_in && (u->key > level))
                  {
                  u_to_w_cost = u->p - a->rev->c + b->c - p;
                  if (u_to_w_cost < 0.0)
                uk = wk;
                  else
#if	defined(STRONG_PO) || !defined(USE_PRICE_OUT)
                {
                /*
                It could happen, in the case of strong price-outs, that
                priced-in arcs cause violation of the condition that node
                price changes are bounded in each iteration. In this case, or
                simply when arc costs are very widely distributed and no
                price-outs are used, some priced-in arcs can have truly huge
                costs, causing the following computation of uk to overflow.
                When this happens, ignore the key. There will be a smaller
                one for the same node.
                */
                if (epsilon * (u->key - wk) > u_to_w_cost)
#endif
                /*
                No ceiling in the following line; such an operation could
                make the gap between a's reduced cost and that of the matching
                arc greater than epsilon, thus violating epsilon optimality.
                */
                uk = wk + 1 + (int64_t) (u_to_w_cost / epsilon);
#if	defined(STRONG_PO) || !defined(USE_PRICE_OUT)
                else uk = u->key;
#endif
                }
                  if (u->key > uk)
                {
                if (u->key != num_buckets)
                  delete_list(u, &bucket[u->key]);
                u->key = uk;
                insert_list(u, &bucket[uk]);
                /* Keep track of to-be-admissible path through this node */
                b->tail->aug_path = b->rev;
                }
                  }
                }
              else
                /*
                Encountered an excess -- b's tail isn't matched. Determine what
                price decrease on b's tail would be required to make the edge to w
                admissible. Recall that back arcs costs are offset so preferred
                arcs have zero partial reduced cost at this point, so we need only
                examine the stored cost of the present edge, rather than compute
                the minimum.
                */
                {
#if	defined(STRONG_PO) || !defined(USE_PRICE_OUT)
                if (epsilon * (closest_dist - wk) > b->c - p)
#endif
                uk = wk + (int64_t) ceil((b->c - p) / epsilon);
#if	defined(STRONG_PO) || !defined(USE_PRICE_OUT)
                else
                  uk = closest_dist;
#endif
                if (uk < closest_dist)
                  {
                  (v = b->tail)->aug_path = b->rev;
                  closest_dist = uk;
                  closest_node = v;
                  if (uk == 0)
                {
#ifdef	DEBUG
                (void) printf("claiming excess at node %ld\n",
                          v - head_lhs_node + 1);
#endif
                break;
                }
                  }
                }

            insert_list(w, &scanned);
            }

            void	sp_aug()

            {
            rhs_ptr	w;
            lhs_ptr	v;
            double	delta_c, this_cost;
            lr_aptr	a, a_stop;
            lhs_ptr	save_active[EXCESS_THRESH];	/* Fix this. It should be */
                                /* malloc'ed once, possibly at */
                                /* full size, and set up by */
                                /* the initialization */
                                /* routines. */
            lhs_ptr	*save_top = save_active,
                *active_node;

            sp_aug_time -= myclock();
            sp_augs++;

#ifdef	DEBUG
            (void) printf("Doing sp_aug(): epsilon = %lg, total_e = %lu\n",
                      epsilon, total_e);
            for (level = 0; level < num_buckets; level++)
              if (bucket[level] != tail_rhs_node)
                 {
                 (void) printf("Bucket init failure!\n");
                 exit(1);
                 }
#endif

            for (level = 0; level < total_e; level++)
              {
              get_active_node(v);
              *(save_top++) = v;
              }

            scanned = tail_rhs_node;
            while (total_e > 0)
              {
              for (active_node = save_active; active_node != save_top; active_node++)
                if (!(v = *active_node)->matched)
                  {
                  v = *active_node;
                  a_stop = (v+1)->priced_out;
#ifdef	DEBUG
                  (void) printf("excess at node %ld\n", v - head_lhs_node + 1);
#endif
                  a = v->first;
#ifdef	NO_FEAS_PROMISE
                  if (a == a_stop)
                {
                (void) printf("Infeasible problem\n");
                exit(9);
                }
#endif
                  delta_c = a->rev->c - a->head->p;
                  for (a++; a != a_stop; a++)
                if ((this_cost = a->rev->c - a->head->p) < delta_c)
                  delta_c = this_cost;
#ifdef	STRONG_PO
                  a_stop = v->priced_out - 1;
#else
                  a_stop = v->first - 1;
#endif
                  for (a--; a != a_stop; a--)
                a->rev->c -= delta_c;
                  }

              /*
              Fix the following so that keys are initialized to the right thing,
              and left there when we're done. Never make the code go through all
              the nodes here. This may mean keeping track of a global list of
              nodes with deficits.
              */
              for (w = head_rhs_node; w != tail_rhs_node; w++)
                if (w->matched)
                  w->key = num_buckets;
                else
                  {
                  w->key = 0;
                  insert_list(w, &bucket[0]);
                  }

              level = 0;
              closest_dist = num_buckets;
              closest_node = tail_lhs_node;

              while (level < closest_dist)
                if (bucket[level] == tail_rhs_node)
                  level++;
                else
                  {
                  w = deq_list(&bucket[level]);
                  a_scan(w);
                  }

#ifdef	DEBUG
              (void) printf("level=%ld, num_buckets=%ld,\n",
                    level, num_buckets);
              (void) printf("closest_dist=%ld, closest_node=%ld\n", closest_dist,
                    closest_node - head_lhs_node + 1);
#endif
              /* Augment from the node with smallest required price change. */
              if (closest_node == tail_lhs_node)
                {
                (void) printf("Error: scanning failure.\n");
                exit(1);
                }
              augment(closest_node);
              while (scanned != tail_rhs_node)
                {
                w = deq_list(&scanned);
                w->p += epsilon * (closest_dist - w->key);
                w->key = num_buckets;
                }
              for (level = closest_dist; level != num_buckets; level++)
                bucket[level] = tail_rhs_node;

              total_e--;
              }

#ifdef	DEBUG
            for (level = 0; level < num_buckets; level++)
              if (bucket[level] != tail_rhs_node)
                 {
                 (void) printf("Bucket check failure!\n");
                 exit(1);
                 }
#endif

            sp_aug_time += myclock();
            }

#endif

///////////////////////////////////////////////////////////////////////////
// sp_aug_forward.c ///////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#if defined(USE_SP_AUG_FORWARD)

            void	augment(rhs_ptr w)

            {
            rhs_ptr	x;
            lhs_ptr	v;
            lr_aptr	a;

            do
              {
              v = w->aug_path->tail;
#ifdef	LOG_PATHS
              (void) printf("%ld %ld", v - head_lhs_node + 1,
                    w - head_rhs_node + 1);
#endif
              w->matched = v;
              a = v->matched;
              v->matched = w->aug_path;
              if (a)
                {
#ifdef	LOG_PATHS
                (void) printf(", ");
#endif
                x = a->head;
                w = x;
                }
              }
            while (a);
#ifdef	LOG_PATHS
            putchar('\n');
#endif
            }

            void	a_scan(w)

            rhs_ptr	w;

            {
            lhs_ptr	v = w->matched;
            rhs_ptr	u;
            lr_aptr	a, a_stop;
            double	delta_c = w->p - v->matched->c,
                w_to_u_cost;
            int64_t	wk = w->key,
                uk;

            a_scans++;

#ifdef	DEBUG
            (void) printf("a_scan(%ld) key=%lu\n", w - head_rhs_node + 1, w->key);
#endif

            a_stop = (v+1)->priced_out;
            for (a = v->first; a != a_stop; a++)
              if (a != v->matched)
                {
                u = a->head;
                w_to_u_cost = delta_c + a->c - u->p;
                if (w_to_u_cost < 0.0)
                  uk = wk;
                else
                  if (epsilon * (u->key - wk) > w_to_u_cost)
                uk = wk + 1 + (int64_t) (w_to_u_cost / epsilon);
                  else
                uk = u->key;
                if (u->key > uk)
                  {
                  if (u->key != num_buckets)
                delete_list(u, &bucket[u->key]);
                  u->key = uk;
                  insert_list(u, &bucket[uk]);
                  u->aug_path = a;
                  }
                }
            }

            /*
            We assume that on entry to sp_aug(), all rhs nodes have their key
            fields set to num_buckets.
            */

            void	sp_aug()

            {
            lhs_ptr	v;
            rhs_ptr	w;
            lr_aptr	a, a_stop;
            size_t	level;
            int64_t	k;
            double	delta_c, this_cost;

            sp_aug_time -= myclock();
            sp_augs++;

#ifdef	DEBUG
            (void) printf("Doing sp_aug(): epsilon = %lg, total_e = %lu\n",
                      epsilon, total_e);
#endif

#ifdef	CHECK_EPS_OPT
              check_e_o(epsilon);
#endif

            while (total_e > 0)
              {
              /*
              Get neighbors of this active node into the proper buckets.
              */
              get_active_node(v);
              a_stop = (v+1)->priced_out;
              a = v->first;
              delta_c = a->c - a->head->p;
              for (a++; a != a_stop; a++)
                if ((this_cost = a->c - a->head->p) < delta_c)
                  delta_c = this_cost;
              a_stop = v->first - 1;
              for (a--; a != a_stop; a--)
                {
                /*
                Insert a's head into the proper bucket with the right key.
                */
                w = a->head;
                w->aug_path = a;
                this_cost = a->c - w->p - delta_c;
                if ((this_cost /= epsilon) < (double) num_buckets)
                  {
                  k = (int64_t) this_cost;
                  if (!w->matched && (k == 0))
                {
                augment(w);
                total_e--;
                break;
                }
                  else if (k < num_buckets)
                {
                /*
                Here we make the (very reasonable) assumption that there are
                no multiple arcs.
                */
                w->key = k;
                insert_list(w, &bucket[k]);
                }
                  }
                }

              level = 0;
              if (a == a_stop)	/* If we didn't find a deficit and augment already */
                {
                while (level < num_buckets)
                  if (bucket[level] == tail_rhs_node)
                level++;
                  else
                {
                w = deq_list(&bucket[level]);
                if (w->matched)
                  a_scan(w);
                else
                  {
                  augment(w);
                  w->key = num_buckets;
                  total_e--;
                  break;
                  }
                }

                if (level == num_buckets)
                  {
                  (void) printf("Error: scanning failure\n");
                  exit(-1);
                  }
                }

              /*
              Adjust prices and clean out remaining buckets.
              */
#ifdef	DEBUG
              (void) printf("level = %lu\n", level);
#endif
              /*
              Nodes at this level and higher need no price adjustment; empty this
              level's bucket pronto.
              */
              bucket[level] = tail_rhs_node;
              /*
              Nodes at levels lower than this need price adjustment; others simply
              need to be deleted from their buckets. This is quicker than going
              through all the buckets in order and cleaning them out.
              */
              for (w = head_rhs_node; w != tail_rhs_node; w++)
                if ((k = w->key) != num_buckets)
                  {
                  if (k < level)
                {
#ifdef	DEBUG
                (void) printf("%ld->p -= %ld * epsilon\n", w - head_rhs_node + 1,
                          level - k);
#endif
                w->p -= (level - k) * epsilon;
                }
                  else if (k > level)
                delete_list(w, &bucket[k]);
                  w->key = num_buckets;
                  }
#ifdef	CHECK_EPS_OPT
              check_e_o(epsilon);
#endif
              }

            sp_aug_time += myclock();
            }

#endif

///////////////////////////////////////////////////////////////////////////
// queue.c ////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#if defined(QUEUE_ORDER)

            queue	q_create(size_t size)

            {
            queue	q;

            q = (queue) malloc(sizeof(struct queue_st));
            if (q == NULL)
              {
              (void) fprintf(stderr,nomem_msg);
              exit(9);
              }
            q->storage = (char **) malloc(sizeof(lhs_ptr) * (size + 1));
            if (q->storage == NULL)
              {
              (void) fprintf(stderr,nomem_msg);
              exit(9);
              }
            q->end = q->storage + size;
            q->tail = q->head = q->storage;
            q->max_size = size;
            return(q);
            }

            char	*deq(queue q)

            {
            char	*p;

            p = *(q->head);
            if (q->head == q->end) q->head = q->storage;
            else q->head++;
            return(p);
            }

            void	q_destroy(queue q)
            {
                free(q->storage);
                free(q);
            }

#endif

///////////////////////////////////////////////////////////////////////////
// list.c /////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#if    defined(USE_P_REFINE) || defined(USE_P_UPDATE) || defined(USE_SP_AUG)

            rhs_ptr	deq_list(rhs_ptr* head)

            {
            rhs_ptr	ans;

            ans = *head;
            *head = ans->next;
            ans->next->prev = tail_rhs_node;
            return(ans);
            }

#endif

///////////////////////////////////////////////////////////////////////////
// p_refine.c /////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#ifdef    USE_P_REFINE

            int	dfs_visit(register rhs_ptr w)

            {
            register	lr_aptr	a, a_stop;
            lhs_ptr	v;
            register	rhs_ptr	x;
            register	double	p;

            w->node_info.srchng = TRUE;
            if (w->node_info.priced_in && (v = w->matched))
              {
              a_stop = (v+1)->priced_out;
              p = w->p - v->matched->c;
              for (a = v->first; a != a_stop; a++)
                if ((a != v->matched) && (p + a->c - (x = a->head)->p < 0.0))
                  {
                  if (x->node_info.srchng)
                return(0);
                  if (!x->node_info.srched && !dfs_visit(x))
                return(0);
                  }
              }
            w->node_info.srchng = FALSE; w->node_info.srched = TRUE;
            st_push(reached_nodes, w);

            return(1);
            }

            int	top_sort()

            {
            register	rhs_ptr	w, w_stop;

            st_reset(reached_nodes);
            for (w = head_rhs_node; w != tail_rhs_node; w++)
              w->node_info.srched = w->node_info.srchng = FALSE;

            w_stop = head_rhs_node - 1;
            for (w--; w != w_stop; w--)
              if (!w->node_info.srched && !dfs_visit(w))
                return(0);
            return(1);
            }

            /*
            Doing an r_scan on w updates the current estimate of required price
            changes on nodes adjacent (in the rhs sense) to w to establish
            epsilon-optimality with no change to the matching.
            */

            void	r_scan(register rhs_ptr w)

            {
            register	lr_aptr	a, a_stop;
            lhs_ptr	v;
            register	rhs_ptr	x;
            register	int64_t	wk, xk;
            register	double	p;
            double	w_to_x_cost;

            r_scans++;
            if (w->node_info.priced_in && (v = w->matched))
              {
              a_stop = (v+1)->priced_out;
              p = w->p - v->matched->c;
              wk = w->key;
              for (a = v->first; a != a_stop; a++)
                if (a != v->matched)
                  {
                  if ((w_to_x_cost = p + a->c - (x = a->head)->p) < 0.0)
                xk = wk;
                  else
                /*
                Preliminary check to avoid overflow and expensive cast operation.
                */
                if ((epsilon * (wk - x->key)) > w_to_x_cost)
                  xk = wk - 1 - (int64_t) (w_to_x_cost / epsilon);
                else
                  xk = x->key;
                  if (xk > x->key)
                {
                delete_list(x, &bucket[x->key]);
                x->key = xk;
                insert_list(x, &bucket[xk]);
                }
                  }
              }
            w->p -= epsilon * w->key;
            w->key = num_buckets;
            }

            int	p_refine()

            {
            register	rhs_ptr	w, x;
            lhs_ptr	v;
            lr_aptr	a, a_stop;
            int64_t	wk, xk, max_key = 0;
            int	eps_opt = FALSE;
            register	double	w_to_x_cost, p;

            p_refine_time -= myclock();
            p_refines++;

            while (top_sort() && !eps_opt)
              {
              for (w = head_rhs_node; w != tail_rhs_node; w++)
                w->key = 0;

              max_key = 0;
              while (!st_empty(reached_nodes))
                {
                w = (rhs_ptr) st_pop(reached_nodes);
                wk = w->key;
                if (wk > max_key) max_key = wk;
                if ((v = w->matched) && w->node_info.priced_in)
                  {
                  a_stop = (v+1)->priced_out;
                  p = w->p - v->matched->c;
                  for (a = v->first; a != a_stop; a++)
                {
                x = a->head;
                if ((epsilon * (wk - x->key)) > (w_to_x_cost = p + a->c - x->p))
                  {
                  xk = wk - 1 - (int64_t) floor(w_to_x_cost / epsilon);
                  if (xk > x->key) x->key = xk;
                  }
                }
                  }
                }

              if (max_key == 0)
                eps_opt = TRUE;
              else
                {
                for (w = head_rhs_node; w != tail_rhs_node; w++)
                  insert_list(w, &bucket[w->key]);
                for (; max_key > 0; max_key--)
                  while (bucket[max_key] != tail_rhs_node)
                r_scan(deq_list(&bucket[max_key]));
                bucket[0] = tail_rhs_node;
                }
              }

            p_refine_time += myclock();
            return(eps_opt);
            }

#endif // USE_P_REFINE

        };


/**
 * Cleanup the mess we made with macros in the header file.
 */

#undef TRUE
#undef FALSE
#undef MAXLINE
#undef DEFAULT_SCALE_FACTOR
#undef DEFAULT_PO_COST_THRESH
#undef DEFAULT_PO_WORK_THRESH
#undef DEFAULT_SCALE_FACTOR
#undef DEFAULT_UPD_FAC
#undef EXCESS_THRESH
#undef PREC_COSTS
#undef NUM_BEST
#undef QUICK_MIN
#undef MIN_COST
#undef EXPLICIT_LHS_PRICES
#undef USE_SP_AUG_BACKWARD
#undef USE_SP_AUG_FORWARD
#undef USE_SP_AUG
#undef USE_P_REFINE
#undef QUEUE_ORDER
#undef ROUND_COSTS
#undef USE_P_UPDATE
#undef STRONG_PO
#undef VERBOSE_TIME
#undef LOOSE_BOUND
#undef BACK_PRICE_OUT
#undef USE_PRICE_OUT
#undef CHECK_EPS_OPT
#undef P_U_ZERO_BACK_MCH_ARCS
#undef NO_FEAS_PROMISE
#undef LOG_PATHS
#undef STORE_REV_ARCS
#undef WORK_TYPE
#undef REFINE_WORK
#undef MAGIC_MARKER
#undef ACTIVE_TYPE
#undef create_active
#undef make_active
#undef get_active_node
#undef st_push
#undef st_empty
#undef enq
#undef q_empty
#undef insert_list
#undef delete_list
#undef copy_lr_arc
#undef copy_rl_arc
#undef exch_rl_arcs
#undef exch_lr_arcs
#undef price_in_rev
#undef price_out_rev
#undef handle_rev_pointers
#undef price_in_unm_arc
#undef price_in_mch_arc
#undef price_out_unm_arc
#undef price_out_mch_arc
#undef ERRBASE
#undef BADINPUT1
#undef BADINPUT2
#undef BADINPUT3
#undef BADINPUT4
#undef BADINPUT5
#undef BADCOUNT
#undef NONCONTIG
#undef NOMEM
#undef sort_insert


    }
}
