"""Subprocess-local GDN selection; the installed vLLM package is untouched."""
from collections import Counter

STATE = dict(variant='baseline', calls=Counter(), shapes=set(), validation=[])


def install(namespace):
    import torch
    from candidate_full import full_fused
    baseline = namespace['chunk_gated_delta_rule_fwd']

    def dispatch(q, k, v, g, beta, scale=None, initial_state=None,
                 output_final_state=False, cu_seqlens=None, chunk_indices=None,
                 chunk_offsets=None, use_qk_l2norm_in_kernel=False,
                 core_attn_out=None):
        # ChunkGatedDeltaRuleFunction.forward has already applied l2norm_fwd.
        kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                      initial_state=initial_state, output_final_state=output_final_state,
                      cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                      chunk_offsets=chunk_offsets, core_attn_out=core_attn_out)
        variant = STATE['variant']
        STATE['calls'][variant] += 1
        STATE['shapes'].add(tuple(q.shape))
        if variant == 'validate':
            # Preserve any in-place output buffer before evaluating the candidate.
            old = baseline(**kwargs)
            expected = [old[1].clone(), old[3].clone()]
            new = full_fused(**kwargs)
            metrics = []
            for actual, reference in zip([new[1], new[3]], expected):
                a, b = actual.float(), reference.float()
                delta = a-b
                maximum = delta.abs().max().item()
                relative = (delta.norm()/b.norm().clamp_min(1e-30)).item()
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                metrics.append(dict(max_abs=maximum, relative_l2=relative, finite=finite))
                if not finite or maximum > 0.02 or relative > 0.01:
                    raise RuntimeError(f'Real-activation GDN parity failed: {metrics}')
            STATE['validation'].append(dict(shape=list(q.shape), checks=metrics))
            # Return the baseline result, including restoration of its output buffer.
            if core_attn_out is not None:
                core_attn_out.copy_(expected[0].view_as(core_attn_out))
            return expected[0], old[3]
        result = (baseline if variant == 'baseline' else full_fused)(**kwargs)
        return result[1], result[3]

    namespace['gdn_test_fwd'] = dispatch
    namespace['E2E_CHUNKED_DISPATCH_INSTALLED'] = True


class GDNExperimentWorker:
    def gdn_e2e_set_variant(self, variant):
        import torch
        from vllm.third_party.flash_linear_attention.ops import chunk
        if variant not in ['baseline', 'fused', 'validate']:
            raise ValueError(variant)
        if not getattr(chunk, 'E2E_CHUNKED_DISPATCH_INSTALLED', False):
            raise RuntimeError('GDN overlay was not installed')
        torch.cuda.synchronize()
        STATE['variant'] = variant
        return self.gdn_e2e_stats()

    def gdn_e2e_stats(self):
        return dict(variant=STATE['variant'], calls=dict(STATE['calls']),
                    shapes=[list(s) for s in sorted(STATE['shapes'])],
                    validation=STATE['validation'])
