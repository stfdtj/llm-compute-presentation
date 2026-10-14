"""Arithmetic audit of the local DeepSeek text-backbone FLOPs report; stdlib only.

Run: python FLOPs_calculation_verify.py
Optional: --history 1023 or --training-batch 2 --training-length 2048.
Reports historical Decode, kernel-revised Decode, and training Scenario P.
Counts modeled operations, not quantization overhead or hardware instructions.
"""
import argparse
import ast
import json
import operator
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def calculate(s=1024):
    c = json.loads((ROOT / 'models/DeepSeek-V4.1-Flash/config.json').read_text())
    source = ast.parse((ROOT / 'models/DeepSeek-V4.1-Flash/model.py').read_text(encoding='utf-8'))
    defaults = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'ModelArgs')
    defaults = {node.target.id: ast.literal_eval(node.value) for node in defaults.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant)}
    assert c.get('gate_temp', defaults['gate_temp']) == 1.0
    assert c.get('norm_topk_prob', defaults['norm_topk_prob']) is True
    assert c.get('temperature', defaults['temperature']) == 1.0
    assert c['n_shared_experts'] == 1 and c['hc_sinkhorn_iters'] == 20
    d, h, hd = c['dim'], c['n_heads'], c['head_dim']
    q, rd, layers = c['q_lora_rank'], c['rope_head_dim'], c['n_layers']
    ih, ik = c['index_n_heads'], c['index_head_dim']
    hc, inter, experts = c['hc_mult'], c['moe_inter_dim'], c['n_activated_experts']
    ratios = c['compress_ratios'][:layers]
    assert (ratios.count(0), ratios.count(2), ratios.count(1)) == (2, 18, 20)
    assert c['kv_source_layers'] == [2, 8, 14, 20]
    assert c['index_source_layers'] == [2, 8, 14, 20, 24, 28, 32, 36]
    assert c['engram_layer_ids'] == [1, 14]
    assert s >= 1
    n = s + 1
    active_index = [i for i in c['index_source_layers'] if n // ratios[i] > 0]
    new_sources = [i for i in c['kv_source_layers'] if n % ratios[i] == 0]
    new_r2 = sum(ratios[i] == 2 for i in new_sources)
    positions = sum(n // ratios[i] for i in active_index)
    selected = sum(min(c['window_size'], n) + (min(c['index_topk'], n // r) if r else 0) for r in ratios)
    ni, nk = len(active_index), len(new_sources)
    # RMSNorm: 4*d multiply/add operations; mean division and rsqrt separate.
    basic = {}
    special = {}
    basic['Q Projection'] = layers * (2*d*q + 2*q*h*hd)
    basic['Q RMSNorm'] = layers*4*q
    special['Q RMSNorm'] = layers*2
    basic['KV Projection + Norm'] = layers*(2*d*hd + 4*hd)
    special['KV Norm'] = layers*2
    basic['Main RoPE'] = layers*(3*h*rd*2 + 3*rd)
    basic['Compressor'] = 7*2*d*hd + nk*(4*hd + 3*rd) + new_r2*6*hd
    special['Compressor'] = nk*2 + new_r2*4*hd
    basic['Indexer'] = ni*(2*q*ih*ik + 2*d*ih + 3*ih*rd + ih) + nk*(2*hd*ik + 4*ik + 3*rd) + (2*ih*ik + 2*ih-1)*positions
    special['Indexer'] = nk*2
    basic['Output Projection'] = layers*(2*(h*hd//c['o_groups'])*c['o_lora_rank']*c['o_groups'] + 2*c['o_groups']*c['o_lora_rank']*d)
    coeff = (2+hc)*hc
    # Norm statistic, GEMM-output scaling, pre, post+comb, 80 instances.
    basic['Hyper-Connection'] = 2*layers*(2*hc*d*coeff + 2*hc*d + coeff + (2*hc-1)*d + hc*(2*hc+1)*d)
    special['Hyper-Connection'] = 2*layers*2
    basic['Attention + FFN Norm'] = 2*layers*4*d
    special['Attention + FFN Norm'] = 2*layers*2
    basic['MoE Router'] = layers*(2*d*c['n_routed_experts'] + c['n_routed_experts'] + (experts-1) + 1 + experts)
    special['MoE Router'] = layers*(3*c['n_routed_experts'] + experts)
    basic['MoE Experts'] = layers*((experts+1)*(6*d*inter + 2*inter) + experts*inter + (experts+1)*d)
    special['MoE Experts'] = layers*(experts+1)*inter
    mem = (c['engram_max_ngram_size']-1)*c['engram_n_heads']*c['engram_head_dim']
    ne = len(c['engram_layer_ids'])
    basic['Engram'] = ne*(mem + 2*mem*d*(hc+1) + hc*(10*d+2))
    special['Engram'] = ne*hc*6
    basic['Final Collapse + Norm + LM Head'] = (2*hc-1)*d + 4*d + 2*d*c['vocab_size']
    special['Final Norm'] = 2
    # Reference attention model: QK + PV, then scale and stable softmax with a zero-value sink.
    attention = {
        'Score GEMM': 2*h*hd*selected,
        'Value GEMM': 2*h*hd*selected,
        'Scale + softmax multiply/add': h*(3*selected+layers),
        'Softmax exp/div': h*(2*selected+layers),
    }
    v = c['vocab_size']
    wrapper = {'Capture multiply/add': len(c['dspark_target_layer_ids'])*(hc-1)*d,
               'Capture mean division': len(c['dspark_target_layer_ids'])*d,
               'Sampling softmax multiply/add': 2*v-1,
               'Sampling div/exp': 4*v}
    return {'S': s, 'N': n, 'new_keys': nk, 'ratio2_emissions': new_r2,
            'index_positions': positions, 'attention_positions': selected,
            'basic': basic, 'special': special, 'attention_estimate': attention,
            'verified_subtotal': sum(basic.values()),
            'special_estimate': sum(special.values()),
            'attention_total_estimate': sum(attention.values()),
            'modeled_backbone': sum(basic.values())+sum(special.values())+sum(attention.values()),
            'wrapper_outside_backbone': wrapper}


def hc_kernel(c=4, iterations=20):
    # Literal source expressions: epsilon addition inside each (j,k) loop.
    basic = 6*c + 2*c*c + (3*c*c-c) + (2*iterations-1)*(2*c*c-c)
    special = 2*c + 2*c*c + (2*iterations-1)*c*c
    return basic, special


def attention_kernel(rows, tile_sum, heads=64, dim=512, tile=64):
    # rows counts query rows across layers, tile_sum counts all 64-position tiles.
    basic = heads*((4*tile*dim + 3*tile+2+dim)*tile_sum + 2*rows)
    special = heads*((tile+1)*tile_sum + (dim+1)*rows)
    return basic, special


def revised_decode(s=1024):
    old = calculate(s)
    keys = {'KV Projection + Norm': 'KV Norm', 'Final Collapse + Norm + LM Head': 'Final Norm'}
    rows = {k: [v, old['special'].get(keys.get(k, k), 0)] for k, v in old['basic'].items()}
    hb, he = hc_kernel()
    rows['Hyper-Connection'][0] += 80*hb
    rows['Hyper-Connection'][1] += 80*he
    n = s+1
    # Actual decode index-list width is 128 even before the ring has filled.
    widths = [128]*2 + [128+min(512,n//2)]*18 + [128+min(512,n)]*20
    tiles = sum((w+63)//64 for w in widths)
    rows['Main Attention'] = list(attention_kernel(40, tiles))
    return rows


def training_forward(batch=1, length=1024):
    """Scenario P: full-sequence source graph, all-token logits; NOT an official trainer."""
    assert batch > 0 and length > 0
    b, t, m = batch, length, batch*length
    rows = {k: [v[0]*m, v[1]*m] for k, v in revised_decode().items()}
    c2, c1 = t//2, t
    emitted = b*(3*c2+c1)
    groups2 = b*3*c2
    rows['Compressor'] = [36700160*m + emitted*(4*512+3*64) + groups2*6*512,
                          2*emitted + groups2*4*512]
    n_index = 5 + (3 if c2 else 0)
    pairs = b*t*(3*c2+5*c1)
    rows['Indexer'] = [n_index*m*(2*1280*4096+2*5120*32+3*32*64+32)
                       + emitted*(2*512*128+4*128+3*64) + (2*32*128+63)*pairs,
                       2*emitted]
    # q_weight*k_weight is parameter-only: one [4,5120] product per layer/call.
    rows['Engram'][0] -= (m-1)*2*4*5120
    window = min(128,t)
    widths = [window]*2 + [window+min(512,c2)]*18 + [window+min(512,c1)]*20
    tiles_per_position = sum((w+63)//64 for w in widths)
    rows['Main Attention'] = list(attention_kernel(40*m, m*tiles_per_position))
    useful_positions = b*sum(40*min(128,i) + 18*min(512,i//2) + 20*min(512,i)
                             for i in range(1,t+1))
    metadata = {'batch': b, 'length': t, 'tokens': m, 'emitted_latents': emitted,
                'index_pairs': pairs, 'tiles_per_position_all_layers': tiles_per_position,
                'attention_padded_positions': m*64*tiles_per_position,
                'attention_useful_positions': useful_positions,
                'useful_attention_gemms': 4*64*512*useful_positions}
    return rows, metadata


def totals(rows):
    basic = sum(v[0] for v in rows.values())
    special = sum(v[1] for v in rows.values())
    return {'basic': basic, 'special': special, 'modeled': basic+special}


def verify_kernel_counts():
    c = json.loads((ROOT / 'models/DeepSeek-V4.1-Flash/config.json').read_text())
    expected = {'dim':5120, 'n_layers':40, 'n_heads':64, 'head_dim':512,
                'q_lora_rank':1280, 'rope_head_dim':64, 'o_groups':8,
                'o_lora_rank':1024, 'index_n_heads':32, 'index_head_dim':128,
                'index_topk':512, 'window_size':128, 'hc_mult':4,
                'hc_sinkhorn_iters':20, 'moe_inter_dim':2304,
                'n_routed_experts':384, 'n_activated_experts':6, 'vocab_size':129280}
    assert all(c[k] == v for k,v in expected.items()), 'Re-derive formulas for changed dimensions'
    kernel = (ROOT / 'models/DeepSeek-V4.1-Flash/kernel.py').read_text(encoding='utf-8-sig')
    ast.parse(kernel)
    assert 'block = 64' in kernel
    assert 'T.serial(sinkhorn_iters - 1)' in kernel
    assert 'acc_o[i, j] *= scores_scale[i]' in kernel
    assert 'acc_o[i, j] /= sum_exp[i]' in kernel
    # Independent loop enumeration of the two source kernels, not the closed forms.
    basic, special = 0, 0
    c, iterations = 4, 20
    for _ in range(c):
        basic += 3+3  # pre affine+epsilon, post affine+multiply
        special += 2
    basic += 2*c*c  # comb affine
    for _ in range(c):
        basic += c+(c-1)+c  # softmax subtraction, sum, epsilon
        special += c+c
    for _ in range(1+2*(iterations-1)):
        for _ in range(c):
            basic += c-1
        for _ in range(c*c):
            basic += 1
            special += 1
    assert (basic, special) == hc_kernel() == (1192,664)
    for tiles in (1,2,3,10):
        basic, special = 0, 0
        for _ in range(64):
            for _ in range(tiles):
                basic += 2*64*512  # QK GEMM, per head
                basic += 64 + 1 + 64 + 63 + 2 + 512
                special += 1 + 64
                basic += 2*64*512  # PV GEMM
            basic += 2
            special += 1+512
        assert (basic,special) == attention_kernel(1,tiles)
    # Training Engram must broadcast one parameter product across all B*T tokens.
    a, _ = training_forward(1,1024)
    b, _ = training_forward(2,1024)
    assert b['Engram'][0] == 2*a['Engram'][0]-40960
    for t in (1,2,3,127,128,129,1023,1024,1025):
        r, meta = training_forward(1,t)
        assert all(v >= 0 for pair in r.values() for v in pair)
        assert meta['attention_useful_positions'] <= meta['attention_padded_positions']
        assert meta['index_pairs'] == t*(3*(t//2)+5*t)


def check_numeric_equations(path):
    """Independently evaluate literal arithmetic lines, without executing report text."""
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.FloorDiv: operator.floordiv, ast.Div: operator.truediv}
    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](value(node.left), value(node.right))
        raise ValueError('Non-numeric equation')
    count = 0
    for line in path.read_text(encoding='utf-8').splitlines():
        match = re.match(r'^\s*([0-9(][0-9 ,.*()+/\-]*)\s*=\s*([0-9][0-9,]*(?:\.[0-9]+)?)(?:\s|$)', line)
        if match:
            expr, expected = (x.replace(',', '') for x in match.groups())
            result = value(ast.parse(expr.strip(), mode='eval').body)
            assert result == float(expected), (line, result)
            count += 1
    return count


def check_summary(result):
    report = (ROOT / 'docs/flops/FLOPs_calculation_forward.txt').read_text(encoding='utf-8-sig')
    rows = {}
    for line in report.splitlines():
        if '|' in line:
            columns = [s.strip() for s in line.split('|')]
            if len(columns) == 3:
                rows[columns[0]] = columns[1:]
    special_by_module = {
        'Q RMSNorm': 'Q RMSNorm', 'KV Projection + Norm': 'KV Norm',
        'Compressor': 'Compressor', 'Indexer': 'Indexer',
        'Hyper-Connection': 'Hyper-Connection', 'Attention + FFN Norm': 'Attention + FFN Norm',
        'MoE Router': 'MoE Router', 'MoE Experts': 'MoE Experts', 'Engram': 'Engram',
        'Final Collapse + Norm + LM Head': 'Final Norm',
    }
    for module, amount in result['basic'].items():
        assert int(rows[module][0].replace(',', '')) == amount, module
        estimate = int(rows[module][1].split('+')[0].strip().replace(',', ''))
        assert estimate == result['special'].get(special_by_module.get(module), 0), module
    assert int(rows['Main Attention'][1].replace(',', '')) == result['attention_total_estimate']
    assert int(rows['TOTAL quantified'][0].replace(',', '')) == result['verified_subtotal']
    assert int(rows['TOTAL quantified'][1].replace(',', '')) == result['special_estimate'] + result['attention_total_estimate']
    audit = (ROOT / 'docs/flops/FLOPs_calculation_audit.md').read_text(encoding='utf-8')
    for field in ('verified_subtotal', 'special_estimate', 'attention_total_estimate', 'modeled_backbone'):
        assert format(result[field], ',') in audit, field
    assert format(calculate(1023)['modeled_backbone'], ',') in audit
    assert format(sum(result['wrapper_outside_backbone'].values()), ',') in audit
    return len(result['basic']) + 2


def check_new_tables():
    report = (ROOT / 'docs/flops/FLOPs_calculation_forward.txt').read_text(encoding='utf-8-sig')
    checked = 0
    for tag, rows in [('D',revised_decode()), ('P',training_forward()[0])]:
        expected = {k: [*v,sum(v)] for k,v in rows.items()}
        result = totals(rows)
        expected['TOTAL'] = [result['basic'],result['special'],result['modeled']]
        for key, value in expected.items():
            prefix = f'{tag} | {key} |'
            matches = [line for line in report.splitlines() if line.startswith(prefix)]
            assert len(matches) == 1, prefix
            actual = [int(s.strip().replace(',','')) for s in matches[0].split('|')[2:5]]
            assert actual == value, (key,actual,value)
            checked += 1
    return checked


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=int, default=1024)
    parser.add_argument('--training-batch', type=int, default=1)
    parser.add_argument('--training-length', type=int, default=1024)
    parser.add_argument('--skip-report-checks', action='store_true')
    args = parser.parse_args()
    result = calculate(args.history)
    if args.history == 1024:
        assert result['index_positions'] == 6661
        assert result['attention_positions'] == 24576
        assert result['new_keys'] == 1
        assert result['basic']['Q Projection'] == 3879731200
        assert result['basic']['Output Projection'] == 6039797760
        report = ROOT / 'docs/flops/FLOPs_calculation_forward.txt'
        if report.exists():
            result['literal_equations_checked'] = check_numeric_equations(report)
        if not args.skip_report_checks:
            result['summary_rows_checked'] = check_summary(result)
            result['revised_summary_rows_checked'] = check_new_tables()
    verify_kernel_counts()
    output = {'historical_decode': result}
    output['revised_decode'] = {'modules': revised_decode(args.history), 'totals': totals(revised_decode(args.history))}
    training, metadata = training_forward(args.training_batch, args.training_length)
    output['training_scenario_P'] = {'modules': training, 'totals': totals(training), 'metadata': metadata,
                                     'average_per_token': totals(training)['modeled']/metadata['tokens']}
    print(json.dumps(output, indent=2))
