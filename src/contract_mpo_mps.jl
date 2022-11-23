function contractmpo_solver(; kwargs...)
  function solver(PH, t, psi; kws...)
    v = ITensor(1.0)
    for j in (PH.lpos + 1):(PH.rpos - 1)
      v *= PH.psi0[j]
    end
    #@show PH.lpos
    #@show PH.rpos
    #@show inds(v)
    Hpsi0 = contract(PH, v)
    #@show inds(Hpsi0)
    return Hpsi0, nothing
  end
  return solver
end

function ITensors.contract(
  ::ITensors.Algorithm"fit", A::MPO, psi0::Union{MPS,MPO}; init_mps=psi0, nsweeps=1, kwargs...
)::MPS
  n = length(A)
  n != length(psi0) &&
    throw(DimensionMismatch("lengths of MPO ($n) and MPS ($(length(psi0))) do not match"))
  if n == 1
    return MPS([A[1] * psi0[1]])
  end

  any(i -> isempty(i), siteinds(commoninds, A, psi0)) &&
    error("In `contract(A::MPO, x::MPS)`, `A` and `x` must share a set of site indices")

  # In case A and psi0 have the same link indices
  A = sim(linkinds, A)

  # Fix site and link inds of init_mps
  init_mps = deepcopy(init_mps)
  init_mps = sim(linkinds, init_mps)
  if init_mps isa MPS
    Ai = siteinds(A)
    ti = Vector{Index}(undef, n)
    for j in 1:n
      for i in Ai[j]
        if !hasind(psi0[j], i)
          ti[j] = i
          break
        end
      end
    end
    replace_siteinds!(init_mps, ti)
  else
    Ai = siteinds(A)
    for n in eachindex(Ai)
      s = noprime(Ai[n][1])
      replaceind!(init_mps[n], s'=>s'')
      @show inds(init_mps[n])
    end
  end

  t = Inf
  reverse_step = false
  PH = ProjMPOApply(psi0, A)
  psi = tdvp(
    contractmpo_solver(; kwargs...), PH, t, init_mps; nsweeps, reverse_step, kwargs...
  )

  return psi
end
