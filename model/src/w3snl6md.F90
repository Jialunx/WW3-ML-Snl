#include "w3macros.h"

MODULE W3SNL6MD

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
  USE ISO_C_BINDING, ONLY: C_INT, C_FLOAT, C_CHAR, C_NULL_CHAR
#endif

  IMPLICIT NONE

#ifdef W3_NL6

  ! C interfaces to ort_wrapper (see ort_wrapper/ort_wrapper.{h,c}).
  INTERFACE
    FUNCTION C_ORT_INIT(model_path) BIND(C, name="ort_init") RESULT(rc)
      IMPORT :: C_INT, C_CHAR
      CHARACTER(KIND=C_CHAR), INTENT(IN) :: model_path(*)
      INTEGER(C_INT)                     :: rc
    END FUNCTION C_ORT_INIT

    FUNCTION C_ORT_FORWARD_4D_F32(input, output, batch, channels, height, width) &
             BIND(C, name="ort_forward_4d_f32") RESULT(rc)
      IMPORT :: C_INT, C_FLOAT
      REAL(C_FLOAT),    INTENT(IN)    :: input(*)
      REAL(C_FLOAT),    INTENT(INOUT) :: output(*)
      INTEGER(C_INT),   VALUE         :: batch, channels, height, width
      INTEGER(C_INT)                  :: rc
    END FUNCTION C_ORT_FORWARD_4D_F32

    SUBROUTINE C_ORT_FINALIZE() BIND(C, name="ort_finalize")
    END SUBROUTINE C_ORT_FINALIZE
  END INTERFACE

  INTEGER, PARAMETER :: NSPEC_TRAIN = 960
  INTEGER, PARAMETER :: NK_TRAIN    = 40
  INTEGER, PARAMETER :: NTH_TRAIN   = 24

  REAL(real32), PARAMETER :: X_STD_IN     = 0.09265626221895218_real32
  REAL(real32), PARAMETER :: Y_STD_GLOBAL = 728.27490234375_real32
  REAL(real32), PARAMETER :: TPIINV_R32   = REAL(TPIINV, real32)
  REAL(real32), PARAMETER :: TPI_R32      = 2.0_real32 * 3.14159265358979_real32
  REAL(real32), PARAMETER :: INV_X_STD_IN = 1.0_real32 / X_STD_IN

  ! 1-based Fortran direction index that the input Q-peak should be rolled to.
  ! Matches Python's dir_center = n_dir // 2 (= 12 for NTH=24 -> Fortran 13).
  INTEGER, PARAMETER :: DIR_CENTER_F = NTH_TRAIN / 2 + 1

  ! Path to the ONNX surrogate model.  At runtime this is taken from the
  ! environment variable WW3_SNL_ONNX_MODEL when set, otherwise the default
  ! below (relative to the run directory) is used.  The bundled weights are
  ! ml_models/unet_faster_24x40_base32_deep.onnx (ML, base-32 deep) and
  ! ml_models/unet_faster_24x40_base16.onnx (ML-Lite, base-16); both share
  ! this single-input depth-scaled module and are selected purely by path.
  CHARACTER(len=*), PARAMETER :: ML_MODEL_DEFAULT = &
       'ml_models/unet_faster_24x40_base32_deep.onnx'

  ! --- Shared state: initialized once at startup, read-only afterwards ---
  !     These are written from INSNL1 (W3IOGR path) in the master thread
  !     before any OMP parallel region, so they can be safely shared.
  INTEGER, SAVE :: ML_IS_INIT      = 0
  INTEGER, SAVE :: FREQ_IS_INIT    = 0
  INTEGER, SAVE :: SIG_IS_INIT     = 0

  REAL(real32), SAVE :: FREQ11(NK_TRAIN)
  REAL(real32), SAVE :: TPIINV_OVER_SIG(NK_TRAIN)
  REAL(real32), SAVE :: SIG_R32(NK_TRAIN)

  ! --- Per-thread state: each OMP thread gets its own copy ---
  !     IN_4D / OUT_4D are the per-point scratch buffers ort_forward
  !     reads from / writes to.  Q_BUF is the un-rolled q grid kept
  !     between ML_NORMALIZE_INPUT and the IN_4D scatter so the
  !     direction roll can be applied in a clean second pass.
  !     ML_GRID_WARNED is threadprivate so the bypass warning is at
  !     worst printed once per thread (harmless).
  INTEGER, SAVE :: ML_GRID_WARNED  = 0

  REAL(real32), SAVE :: IN_4D(1,1,NTH_TRAIN,NK_TRAIN)
  REAL(real32), SAVE :: OUT_4D(1,1,NTH_TRAIN,NK_TRAIN)
  REAL(real32), SAVE :: Q_BUF(NTH_TRAIN, NK_TRAIN)
  REAL(real32), SAVE :: CONX_IFR(NK_TRAIN)

  !$OMP THREADPRIVATE(ML_GRID_WARNED,                          &
  !$OMP&              IN_4D, OUT_4D, Q_BUF, CONX_IFR)

  ! Shared (non-threadprivate) buffers used only for the one-shot
  ! warm-up forward in ML_ENSURE_READY.  Running one forward before the
  ! integration loop populates the ORT graph optimiser caches and
  ! arena allocator out of the Elapsed-time window.
  REAL(real32), SAVE :: WARMUP_IN(1,1,NTH_TRAIN,NK_TRAIN)
  REAL(real32), SAVE :: WARMUP_OUT(1,1,NTH_TRAIN,NK_TRAIN)

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

  SUBROUTINE ML_INIT_FREQ()
    IMPLICIT NONE
    INTEGER :: n
    REAL(real32) :: fcur

    IF (FREQ_IS_INIT == 1) RETURN

    fcur = 0.03453_real32
    FREQ11(1) = fcur ** 11

    DO n = 2, NK_TRAIN
      fcur = fcur * 1.1_real32
      FREQ11(n) = fcur ** 11
    END DO

    FREQ_IS_INIT = 1
  END SUBROUTINE ML_INIT_FREQ


  SUBROUTINE ML_INIT_SIG()
    IMPLICIT NONE
    INTEGER :: n

    IF (SIG_IS_INIT == 1) RETURN

    DO n = 1, NK_TRAIN
      TPIINV_OVER_SIG(n) = TPIINV_R32 / REAL(SIG(n), real32)
      SIG_R32(n)         = REAL(SIG(n), real32)
    END DO

    SIG_IS_INIT = 1
  END SUBROUTINE ML_INIT_SIG


  PURE REAL(real32) FUNCTION ML_DIA_DEPTH_SCALE(KD_EFF)
    IMPLICIT NONE
    REAL(real32), INTENT(IN) :: KD_EFF
    REAL(real32) :: x2dia
    !
    ! Hasselmann-style finite-depth correction R(kd) used by the training
    ! pipeline's compute_R_with_limiter.  Caller supplies kd_eff already
    ! limited from below by 0.5 (matches kdp_min in the Python).
    !
    x2dia = MAX(-1.0e15_real32, -1.25_real32 * KD_EFF)

    ML_DIA_DEPTH_SCALE = 1.0_real32 + 5.5_real32 / KD_EFF * &
                         (1.0_real32 - 0.83333333_real32 * KD_EFF) * EXP(x2dia)
  END FUNCTION ML_DIA_DEPTH_SCALE


  SUBROUTINE ML_ENSURE_READY()
    !
    ! One-time shared initialization.  Called from INSNL1 (W3IOGR) during
    ! grid setup, before any OMP parallel region.  Only touches shared
    ! state: the frequency / sigma tables and the torch model handle.
    !
    ! The per-thread input / output tensor wrappers are bound lazily from
    ! ML_SNL_FORWARD via ML_ENSURE_THREAD_TENSORS, so each OMP thread gets
    ! its own wrappers pointing at its own threadprivate IN_4D / OUT_4D
    ! scratch buffers.
    !
    IMPLICIT NONE
    CHARACTER(len=1024) :: model_path
    INTEGER             :: env_len, env_stat

    CALL ML_INIT_FREQ()
    CALL ML_INIT_SIG()

    IF (ML_IS_INIT == 0) THEN
      ! Resolve the model path from the environment, falling back to the
      ! bundled default.  Lets ML deep / ML-Lite be swapped without rebuild.
      CALL GET_ENVIRONMENT_VARIABLE('WW3_SNL_ONNX_MODEL', model_path, &
                                    env_len, env_stat)
      IF (env_stat /= 0 .OR. env_len == 0) model_path = ML_MODEL_DEFAULT
      IF (C_ORT_INIT(TRIM(model_path) // C_NULL_CHAR) /= 0) THEN
        WRITE(*,*) 'W3SNL6: ort_init failed for ', TRIM(model_path)
        STOP 1
      END IF
      ML_IS_INIT = 1

      ! Warm-up: do one forward pass on a zero input so the ORT graph
      ! optimiser caches and arena allocator are populated here, in
      ! init, rather than on the first real W3SNL6 call inside the
      ! integration loop.
      WARMUP_IN  = 0.0_real32
      WARMUP_OUT = 0.0_real32
      IF (C_ORT_FORWARD_4D_F32(WARMUP_IN, WARMUP_OUT, &
                               1, 1, NTH_TRAIN, NK_TRAIN) /= 0) THEN
        WRITE(*,*) 'W3SNL6: ort_forward warm-up failed'
        STOP 1
      END IF
    END IF
  END SUBROUTINE ML_ENSURE_READY


  SUBROUTINE ML_ENSURE_THREAD_TENSORS()
    !
    ! No-op for the ONNX backend.  The C wrapper creates fresh OrtValue
    ! tensors that borrow IN_4D / OUT_4D directly on every forward call,
    ! so there is no per-thread tensor wrapper state to bind.  Kept as
    ! an empty entry point so callers don't have to be conditional.
    !
    IMPLICIT NONE
  END SUBROUTINE ML_ENSURE_THREAD_TENSORS


  SUBROUTINE ML_NORMALIZE_INPUT(A, CG, QMAX, SHIFT, F_IDX_PK)
    !
    ! Mirror of the Python training transform (normalize_dataset, center_dir=True):
    !
    !   G(theta, f) = A * 2*pi*sigma / cg               (= efth, the trained spectrum)
    !   Q(theta, f) = f^11 * G^3
    !   (k_idx, f_idx) = argmax(Q)                      ! global 2D argmax
    !   F_n = G[k_idx, f_idx],  f_n = f[f_idx],  qmax = Q[k_idx, f_idx] = F_n^3 * f_n^11
    !   x_norm = Q / qmax
    !   x_centered = roll(x_norm, shift = DIR_CENTER - k_idx, axis = direction)
    !   IN_4D = x_centered / X_STD_IN
    !
    ! Returns qmax (= F_n^3 * f_n^11), the centering SHIFT, and the peak
    ! frequency index F_IDX_PK so the caller can derive R(kdp) and run the
    ! inverse transform.  CONX_IFR(IFR) = cg/(2*pi*sigma) is filled as a
    ! side effect so ML_DENORMALIZE_OUTPUT can re-apply the ounp pre-factor.
    !
    IMPLICIT NONE
    REAL, INTENT(IN)             :: A(NSPEC), CG(NK)
    REAL(real32), INTENT(OUT)    :: QMAX
    INTEGER, INTENT(OUT)         :: SHIFT, F_IDX_PK

    INTEGER      :: IFR, ITH, ISP, K_IDX_PK, TARGET_ITH
    REAL(real32) :: cg_r32, conx, inv_conx, ue, q, qscale

    QMAX     = -1.0_real32
    K_IDX_PK = 1
    F_IDX_PK = 1

    DO IFR = 1, NK
      cg_r32   = REAL(CG(IFR), real32)
      conx     = TPIINV_OVER_SIG(IFR) * cg_r32
      inv_conx = 1.0_real32 / conx
      CONX_IFR(IFR) = conx

      DO ITH = 1, NTH
        ISP = ITH + (IFR - 1) * NTH

        ue = REAL(A(ISP), real32) * inv_conx
        q  = FREQ11(IFR) * ue * ue * ue

        Q_BUF(ITH, IFR) = q

        IF (q > QMAX) THEN
          QMAX     = q
          K_IDX_PK = ITH
          F_IDX_PK = IFR
        END IF
      END DO
    END DO

    IF (QMAX <= 0.0_real32) THEN
      SHIFT = 0
      RETURN
    END IF

    ! Direction shift to roll the Q-peak to DIR_CENTER_F.  MODULO keeps it
    ! in [0, NTH-1] so the per-ITH wrap-around is a single integer add.
    SHIFT  = MODULO(DIR_CENTER_F - K_IDX_PK, NTH)
    qscale = INV_X_STD_IN / QMAX

    DO IFR = 1, NK
      DO ITH = 1, NTH
        TARGET_ITH = MODULO(ITH - 1 + SHIFT, NTH) + 1
        IN_4D(1, 1, TARGET_ITH, IFR) = Q_BUF(ITH, IFR) * qscale
      END DO
    END DO

  END SUBROUTINE ML_NORMALIZE_INPUT


  SUBROUTINE ML_DEPTH_SCALE_FROM_FPEAK(DEPTH_IN, F_IDX_PK, DEPTH_SCALE)
    !
    ! R from peak-frequency wavenumber, matching the Python:
    !   compute_kdp_from_spectrum -> kp = solve_k(omega_p, depth)
    !   compute_R_with_limiter    -> kdp_eff = max(0.5, kp * depth)
    !                                R = 1 + 5.5/kdp_eff * (1 - 5/6*kdp_eff)
    !                                       * exp(-5/4*kdp_eff)
    ! Newton on sigma^2 = g * k * tanh(k*d), 30 iterations, deep-water guess.
    !
    IMPLICIT NONE
    REAL(real32), INTENT(IN)  :: DEPTH_IN
    INTEGER, INTENT(IN)       :: F_IDX_PK
    REAL(real32), INTENT(OUT) :: DEPTH_SCALE

    INTEGER      :: niter
    REAL(real32) :: omega, kk, kd_iter, tanh_kd, fval, dfval, kp_r32, kd_eff

    omega = SIG_R32(F_IDX_PK)
    kk    = omega * omega / 9.81_real32

    DO niter = 1, 30
      kd_iter = kk * DEPTH_IN
      tanh_kd = TANH(kd_iter)
      fval    = 9.81_real32 * kk * tanh_kd - omega * omega
      dfval   = 9.81_real32 * (tanh_kd + kd_iter * (1.0_real32 - tanh_kd * tanh_kd))
      kk      = kk - fval / MAX(dfval, 1.0e-30_real32)
    END DO

    kp_r32      = MAX(kk, 1.0e-12_real32)
    kd_eff      = MAX(0.5_real32, kp_r32 * DEPTH_IN)
    DEPTH_SCALE = ML_DIA_DEPTH_SCALE(kd_eff)
  END SUBROUTINE ML_DEPTH_SCALE_FROM_FPEAK


  SUBROUTINE ML_DENORMALIZE_OUTPUT(QMAX, SHIFT, DEPTH_SCALE, S)
    !
    ! Inverse of the Python output transform (denorm_y_batch in normalize_dataset):
    !
    !   y_norm  = OUT_4D                                  (still centered)
    !   y_unc   = roll(y_norm * y_scale, -SHIFT, axis = direction)
    !   S_phys  = y_unc * R * F_n^3 * f_n^11
    !           = OUT_4D * Y_STD_GLOBAL * R * QMAX        (since qmax = F_n^3 * f_n^11)
    !
    ! The caller (w3srcemd / ww3_ounp) then multiplies by FACTOR = 2*pi*sigma/cg
    ! before storing as 'snl', so we pre-divide by FACTOR here, i.e. multiply
    ! by CONX_IFR(IFR) = cg/(2*pi*sigma) for each frequency band.
    !
    ! Inverse direction roll uses the same SHIFT as the forward (so the
    ! shift cancels): physical S at ITH reads OUT_4D at index
    !   SRC_ITH = MODULO((ITH - 1) + SHIFT, NTH) + 1.
    !
    IMPLICIT NONE
    REAL(real32), INTENT(IN) :: QMAX, DEPTH_SCALE
    INTEGER, INTENT(IN)      :: SHIFT
    REAL, INTENT(OUT)        :: S(NSPEC)

    INTEGER      :: IFR, ITH, ISP, SRC_ITH
    REAL(real32) :: den_fac

    den_fac = QMAX * DEPTH_SCALE * Y_STD_GLOBAL

    DO IFR = 1, NK
      DO ITH = 1, NTH
        ISP     = ITH + (IFR - 1) * NTH
        SRC_ITH = MODULO(ITH - 1 + SHIFT, NTH) + 1
        S(ISP)  = REAL(OUT_4D(1, 1, SRC_ITH, IFR) * den_fac * CONX_IFR(IFR), &
                       KIND(S(ISP)))
      END DO
    END DO
  END SUBROUTINE ML_DENORMALIZE_OUTPUT


  SUBROUTINE ML_SNL_FORWARD(A, CG, DEPTH_IN, S, D, TAG)
    !
    ! Orchestrates the per-point ML evaluation:
    !   1) ML_NORMALIZE_INPUT     -> IN_4D in centered, scaled space
    !   2) torch_model_forward    -> OUT_4D (still in centered space)
    !   3) ML_DEPTH_SCALE_FROM_FPEAK -> R(kp * depth)
    !   4) ML_DENORMALIZE_OUTPUT  -> S in WW3 pre-multiplied units
    !
    IMPLICIT NONE

    REAL, INTENT(IN)  :: A(NSPEC), CG(NK), DEPTH_IN
    REAL, INTENT(OUT) :: S(NSPEC), D(NSPEC)
    CHARACTER(len=*), INTENT(IN) :: TAG

    INTEGER      :: SHIFT, F_IDX_PK
    REAL(real32) :: qmax, depth_scale, depth_r32

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

    ! Bind this thread's torch tensor wrappers to its own IN_4D / OUT_4D
    ! buffers on the first call.  Cheap on subsequent calls.
    CALL ML_ENSURE_THREAD_TENSORS()

    CALL ML_NORMALIZE_INPUT(A, CG, qmax, SHIFT, F_IDX_PK)

    ! qmax <= 0 covers both the all-zero spectrum and any numerical edge
    ! case; torch_model_forward is skipped in that path.
    IF (qmax <= 0.0_real32) THEN
      S(:) = 0.0
      RETURN
    END IF

    IF (C_ORT_FORWARD_4D_F32(IN_4D, OUT_4D, 1, 1, NTH_TRAIN, NK_TRAIN) /= 0) THEN
      WRITE(*,*) 'W3SNL6: ort_forward failed'
      STOP 1
    END IF

    depth_r32 = REAL(DEPTH_IN, real32)
    CALL ML_DEPTH_SCALE_FROM_FPEAK(depth_r32, F_IDX_PK, depth_scale)

    CALL ML_DENORMALIZE_OUTPUT(qmax, SHIFT, depth_scale, S)

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
