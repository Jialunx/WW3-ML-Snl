#include "w3macros.h"

MODULE W3SNL6MD
  !
  ! ML-FiLM finite-depth S_nl operator (NL6).
  !
  ! Unlike the deep-scaled NL6 (build_ml_b32deep), which fed a single-input
  ! deep-water U-Net and applied a post-hoc scalar Hasselmann R(kd), this
  ! version drives the depth-conditional FiLM surrogate.  The ENTIRE inference
  ! recipe (kdp from the directionally-integrated peak, R-limiter, y_amp, the
  ! Q = f^11 G^3 peak centering, log1p input/expm1 output normalization, the
  ! 42-dim group-velocity condition vector and its z-scoring, the U-Net forward)
  ! is wrapped inside ONE TorchScript module:
  !
  !     S_efth(theta,f) = FiLM( efth(theta,f) , depth )
  !
  ! so the Fortran only has to (1) build efth = A * 2*pi*sigma / cg, (2) supply
  ! the scalar depth, (3) call the two-input forward, and (4) rescale the
  ! returned S back to WW3 action units by conx = cg / (2*pi*sigma).
  !
  ! The wrapped module was validated to reproduce the offline R.ml_snl recipe
  ! to ~0% NRMSE across depths 2-3000 m before being exported.
  !
  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: real32
  USE CONSTANTS,  ONLY : TPIINV
  USE W3GDATMD,   ONLY : NK, NTH, NSPEC, SIG, KDCON, KDMN, SNLS1, SNLS2, SNLS3

#ifdef W3_T0
  USE W3ODATMD,   ONLY : NDST
  USE W3ARRYMD,   ONLY : PRT2DS
#endif

#ifdef W3_T1
  USE W3ODATMD,   ONLY : NDST
  USE W3ARRYMD,   ONLY : OUTMAT
#endif

#ifdef W3_NL6
  USE, INTRINSIC :: ISO_C_BINDING, ONLY: C_INT, C_FLOAT, C_CHAR, C_NULL_CHAR
#endif

  IMPLICIT NONE

#ifdef W3_NL6

  INTEGER, PARAMETER :: NSPEC_TRAIN = 960
  INTEGER, PARAMETER :: NK_TRAIN    = 40
  INTEGER, PARAMETER :: NTH_TRAIN   = 24

  REAL(real32), PARAMETER :: TPIINV_R32 = REAL(TPIINV, real32)

  ! Path to the depth-conditioned (FiLM) ONNX model.  Taken from the
  ! environment variable WW3_SNL_ONNX_MODEL when set, otherwise the bundled
  ! default below (relative to the run directory) is used.
  CHARACTER(len=*), PARAMETER :: ML_MODEL_DEFAULT = &
       'ml_models/cond_unet_film_24x40.onnx'

  ! --- Shared state: initialized once at startup, read-only afterwards ---
  INTEGER, SAVE :: ML_IS_INIT   = 0
  INTEGER, SAVE :: SIG_IS_INIT  = 0
  ! ONNX Runtime C interface (ort_wrapper_film)
  INTERFACE
    FUNCTION C_ORT_INIT(model_path) BIND(C, name="ort_init") RESULT(rc)
      IMPORT :: C_INT, C_CHAR
      CHARACTER(KIND=C_CHAR), INTENT(IN) :: model_path(*)
      INTEGER(C_INT) :: rc
    END FUNCTION C_ORT_INIT
    FUNCTION C_ORT_FORWARD_FILM(efth, depth, output, batch, channels, &
                                height, width) &
             BIND(C, name="ort_forward_film") RESULT(rc)
      IMPORT :: C_INT, C_FLOAT
      REAL(C_FLOAT),  INTENT(IN)    :: efth(*), depth(*)
      REAL(C_FLOAT),  INTENT(INOUT) :: output(*)
      INTEGER(C_INT), VALUE         :: batch, channels, height, width
      INTEGER(C_INT) :: rc
    END FUNCTION C_ORT_FORWARD_FILM
    SUBROUTINE C_ORT_FINALIZE() BIND(C, name="ort_finalize")
    END SUBROUTINE C_ORT_FINALIZE
  END INTERFACE

  ! conx pre-factor table  conx(f) = cg/(2*pi*sigma) ; here only the
  ! 1/(2*pi*sigma) part is precomputed (cg is per-call, depth dependent).
  REAL(real32), SAVE :: TPIINV_OVER_SIG(NK_TRAIN)

  ! --- Per-thread state ---
  !     IN_4D (efth), DEP_2D (depth), OUT_4D (S) are the per-point scratch
  !     buffers the torch tensor wrappers point at, so both the buffers and
  !     their wrappers must live in the same thread.  CONX_IFR keeps the
  !     per-frequency conx factor between ML_FILL_EFTH and ML_RESCALE_OUTPUT.
  INTEGER, SAVE :: ML_TENS_IS_INIT = 0
  INTEGER, SAVE :: ML_GRID_WARNED  = 0

  ! ONNX backend: the ORT wrapper owns its OrtValue tensors; no Fortran ones.

  REAL(real32), SAVE :: IN_4D(1,1,NTH_TRAIN,NK_TRAIN)
  REAL(real32), SAVE :: OUT_4D(1,1,NTH_TRAIN,NK_TRAIN)
  REAL(real32), SAVE :: DEP_2D(1,1)
  REAL(real32), SAVE :: CONX_IFR(NK_TRAIN)

  !$OMP THREADPRIVATE(ML_TENS_IS_INIT, ML_GRID_WARNED,         &
  !$OMP&              IN_4D, OUT_4D, DEP_2D, CONX_IFR)

#endif

CONTAINS

  SUBROUTINE INSNL1(IGRD)
    IMPLICIT NONE
    INTEGER, INTENT(IN) :: IGRD
#ifdef W3_NL6
    CALL ML_ENSURE_READY()
#endif
    RETURN
  END SUBROUTINE INSNL1


  SUBROUTINE INSNLGQM
    IMPLICIT NONE
    RETURN
  END SUBROUTINE INSNLGQM


  SUBROUTINE W3SNL6(A, CG, DEPTH, S, D)
    IMPLICIT NONE
    REAL, INTENT(IN)  :: A(NSPEC), CG(NK), DEPTH
    REAL, INTENT(OUT) :: S(NSPEC), D(NSPEC)
#ifdef W3_NL6
    CALL ML_SNL_FORWARD(A, CG, DEPTH, S, D, 'W3SNL6')
#else
    S(:) = 0.0
    D(:) = 0.0
#endif
    RETURN
  END SUBROUTINE W3SNL6


#ifdef W3_NL6

  SUBROUTINE ML_INIT_SIG()
    !
    ! Precompute 1/(2*pi*sigma) per frequency.  conx(f) = this * cg(f).
    !
    IMPLICIT NONE
    INTEGER :: n

    IF (SIG_IS_INIT == 1) RETURN

    DO n = 1, NK_TRAIN
      TPIINV_OVER_SIG(n) = TPIINV_R32 / REAL(SIG(n), real32)
    END DO

    SIG_IS_INIT = 1
  END SUBROUTINE ML_INIT_SIG


  SUBROUTINE ML_ENSURE_READY()
    !
    ! One-time shared initialization.  Called from INSNL1 (W3IOGR) during grid
    ! setup, before any OMP parallel region.  Loads the FiLM TorchScript model
    ! and fills the shared sigma table.  Per-thread tensor wrappers are bound
    ! lazily from ML_SNL_FORWARD.
    !
    IMPLICIT NONE
    CHARACTER(len=1024) :: model_path
    INTEGER             :: env_len, env_stat

    CALL ML_INIT_SIG()

    IF (ML_IS_INIT == 0) THEN
      CALL GET_ENVIRONMENT_VARIABLE('WW3_SNL_ONNX_MODEL', model_path, &
                                    env_len, env_stat)
      IF (env_stat /= 0 .OR. env_len == 0) model_path = ML_MODEL_DEFAULT
      IF (C_ORT_INIT(TRIM(model_path) // C_NULL_CHAR) /= 0) THEN
        WRITE(*,*) 'W3SNL6: ort_init failed for ', TRIM(model_path)
        ML_IS_INIT = 0
        RETURN
      END IF
      ML_IS_INIT = 1
    END IF
  END SUBROUTINE ML_ENSURE_READY


  SUBROUTINE ML_ENSURE_THREAD_TENSORS()
    !
    ! Per-thread lazy binding of the torch tensor wrappers to this thread's own
    ! IN_4D (efth), DEP_2D (depth) and OUT_4D (S) scratch buffers.  The input
    ! array has TWO tensors: (1) efth, (2) depth, matching the wrapped module's
    ! forward(efth, depth).
    !
    IMPLICIT NONE

    ! ONNX backend: nothing per-thread to bind (ORT wrapper owns its tensors)
    ML_TENS_IS_INIT = 1
  END SUBROUTINE ML_ENSURE_THREAD_TENSORS


  SUBROUTINE ML_FILL_EFTH(A, CG, EMAX)
    !
    ! Build the energy density spectrum efth(theta,f) = A * 2*pi*sigma / cg
    ! into IN_4D (the wrapper's first input), and store conx(f) = cg/(2*pi*sigma)
    ! in CONX_IFR for the inverse rescale.  Returns the spectrum maximum so the
    ! caller can skip the forward on an all-zero spectrum.
    !
    IMPLICIT NONE
    REAL, INTENT(IN)          :: A(NSPEC), CG(NK)
    REAL(real32), INTENT(OUT) :: EMAX

    INTEGER      :: IFR, ITH, ISP
    REAL(real32) :: cg_r32, conx, inv_conx, ue

    EMAX = 0.0_real32

    DO IFR = 1, NK
      cg_r32        = REAL(CG(IFR), real32)
      conx          = TPIINV_OVER_SIG(IFR) * cg_r32     ! cg/(2*pi*sigma)
      inv_conx      = 1.0_real32 / conx                 ! 2*pi*sigma/cg
      CONX_IFR(IFR) = conx

      DO ITH = 1, NTH
        ISP = ITH + (IFR - 1) * NTH
        ue  = REAL(A(ISP), real32) * inv_conx           ! efth
        IN_4D(1, 1, ITH, IFR) = ue
        IF (ue > EMAX) EMAX = ue
      END DO
    END DO
  END SUBROUTINE ML_FILL_EFTH


  SUBROUTINE ML_RESCALE_OUTPUT(S)
    !
    ! The wrapped module returns S in efth (energy-density) space.  WW3 wants
    ! the action-space source, so multiply by conx(f) = cg/(2*pi*sigma); the
    ! downstream ww3_ounp 'snl' pre-factor 2*pi*sigma/cg then cancels it back to
    ! efth space for output, exactly as for the WRT (NL2) operator.
    !
    IMPLICIT NONE
    REAL, INTENT(OUT) :: S(NSPEC)

    INTEGER :: IFR, ITH, ISP

    DO IFR = 1, NK
      DO ITH = 1, NTH
        ISP    = ITH + (IFR - 1) * NTH
        S(ISP) = REAL(OUT_4D(1, 1, ITH, IFR) * CONX_IFR(IFR), KIND(S(ISP)))
      END DO
    END DO
  END SUBROUTINE ML_RESCALE_OUTPUT


  SUBROUTINE ML_SNL_FORWARD(A, CG, DEPTH_IN, S, D, TAG)
    !
    ! Per-point FiLM evaluation:
    !   1) ML_FILL_EFTH        -> IN_4D = efth, CONX_IFR, EMAX
    !   2) DEP_2D = depth
    !   3) ORT forward(efth, depth) -> S_efth
    !   4) ML_RESCALE_OUTPUT   -> S = S_efth * conx
    !
    IMPLICIT NONE

    REAL, INTENT(IN)  :: A(NSPEC), CG(NK), DEPTH_IN
    REAL, INTENT(OUT) :: S(NSPEC), D(NSPEC)
    CHARACTER(len=*), INTENT(IN) :: TAG

    REAL(real32) :: emax

#ifdef W3_T0
    REAL :: SOUT(NK, NTH), DOUT(NK, NTH)
    INTEGER :: IFR, ITH, ISP
#endif

    D(:) = 0.0

    IF (NSPEC /= NSPEC_TRAIN .OR. NK /= NK_TRAIN .OR. NTH /= NTH_TRAIN) THEN
      IF (ML_GRID_WARNED == 0) THEN
        WRITE(*,'(A,1X,A,1X,A,3(1X,I0),A,3(1X,I0))') &
          'ML-SNL BYPASS,', TRIM(TAG), ': grid=', NSPEC, NK, NTH, &
          ' expected=', NSPEC_TRAIN, NK_TRAIN, NTH_TRAIN
        ML_GRID_WARNED = 1
      END IF
      S(:) = 0.0
      RETURN
    END IF

    IF (ML_IS_INIT == 0) THEN
      S(:) = 0.0
      RETURN
    END IF

    CALL ML_ENSURE_THREAD_TENSORS()

    CALL ML_FILL_EFTH(A, CG, emax)

    IF (emax <= 0.0_real32) THEN
      S(:) = 0.0
      RETURN
    END IF

    DEP_2D(1, 1) = REAL(DEPTH_IN, real32)

    IF (C_ORT_FORWARD_FILM(IN_4D, DEP_2D, OUT_4D, 1, 1, &
                           NTH_TRAIN, NK_TRAIN) /= 0) THEN
      S(:) = 0.0
      RETURN
    END IF

    CALL ML_RESCALE_OUTPUT(S)

#ifdef W3_T0
    DO IFR = 1, NK
      DO ITH = 1, NTH
        ISP           = ITH + (IFR - 1) * NTH
        SOUT(IFR,ITH) = S(ISP)
        DOUT(IFR,ITH) = D(ISP)
      END DO
    END DO

    CALL PRT2DS(NDST, NK, NK, NTH, SOUT, SIG(1:), '  ', 1.0, &
         0.0, 0.001, 'Snl(f,t)', ' ', 'NONAME')
    CALL PRT2DS(NDST, NK, NK, NTH, DOUT, SIG(1:), '  ', 1.0, &
         0.0, 0.001, 'Diag Snl', ' ', 'NONAME')
#endif

#ifdef W3_T1
    CALL OUTMAT(NDST, S, NTH, NTH, NK, 'Snl')
    CALL OUTMAT(NDST, D, NTH, NTH, NK, 'Diag Snl')
#endif

    RETURN
  END SUBROUTINE ML_SNL_FORWARD


  SUBROUTINE ML_SNL_FINALIZE()
    IMPLICIT NONE

    ML_TENS_IS_INIT = 0
    IF (ML_IS_INIT == 1) THEN
      CALL C_ORT_FINALIZE()
      ML_IS_INIT = 0
    END IF
  END SUBROUTINE ML_SNL_FINALIZE

#endif

END MODULE W3SNL6MD


SUBROUTINE W3SNLGQM(A, CG, DEPTH, S, D)
  USE W3SNL6MD, ONLY: ML_SNL_FORWARD
  USE W3GDATMD, ONLY: NSPEC, NK
  IMPLICIT NONE

  REAL, INTENT(IN)  :: A(NSPEC)
  REAL, INTENT(IN)  :: CG(NK)
  REAL, INTENT(IN)  :: DEPTH
  REAL, INTENT(OUT) :: S(NSPEC)
  REAL, INTENT(OUT) :: D(NSPEC)

#ifdef W3_NL6
  CALL ML_SNL_FORWARD(A, CG, DEPTH, S, D, 'W3SNLGQM')
#else
  S(:) = 0.0
  D(:) = 0.0
#endif

  RETURN
END SUBROUTINE W3SNLGQM
