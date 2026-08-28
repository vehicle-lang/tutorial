From mathcomp Require Import ssreflect all_boot all_algebra order reals lra.
From mathcomp Require Import interval_inference tensor.
From vehicle Require Import utils.
Import Num.Theory GRing.Theory Order.POrderTheory.

Local Open Scope ring_scope.
Local Open Scope order_scope.

From centredStability Require Import WindControllerSpec.

Import WindControllerSpec.

Notation R := WindControllerSpec.R.

(* Vehicle 0.26.1 needs an explicit bound for every network input.  The VCL
   solution therefore writes |previousSensor| <= 0.4 when invoking the
   generated lemma.  This is not a third physical assumption: the theorem
   derives it from the two stability assumptions before applying the lemma. *)
Theorem controllerCenterBound :
  forall x, stableNearCenterInput x -> smallControllerOutput x.
Proof.
move=> x [[currentLower currentUpper] [changeLower changeUpper]].
apply sensorNearCenter; split.
  by split.
rewrite /derivedPreviousSensorBound; split.
- apply/(proj1 (tensor_nil_leP _ _)).
  rewrite rmorphN.
  move: (proj2 (tensor_nil_leP _ _) currentLower).
  move: (proj2 (tensor_nil_leP _ _) changeUpper).
  rewrite !rmorphN !raddfB !const_tK.
  lra.
- apply/(proj1 (tensor_nil_leP _ _)).
  move: (proj2 (tensor_nil_leP _ _) currentUpper).
  move: (proj2 (tensor_nil_leP _ _) changeLower).
  rewrite !rmorphN !raddfB !const_tK.
  lra.
Qed.
