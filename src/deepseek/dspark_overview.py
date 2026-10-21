"""Connected DSpark teaching branch; confidence is a separate data path."""
import build_diagram as d

def topic(key,x,y,w,h,lines,size=11):
    d.raw(f'<g class="dspark-topic" data-dspark-topic="{key}" role="button" tabindex="0" aria-label="DSpark {key}">')
    d.rect(x,y,w,h,'purple')
    d.text(x+w/2,y+h/2-(len(lines)-1)*size*.55+size*.33,lines,size,leading=size*1.1,maxwidth=w-8)
    d.raw('</g>')

def link(key,path,dashed=False):
    d.raw(f'<g data-dspark-link="{key}">')
    d.arrow(path,'#8d4bc6',dashed,1.4)
    d.raw('</g>')

def draw():
    d.raw('<g id="dspark-overview" font-family="Times New Roman, Microsoft YaHei, serif">')
    d.rect(900,87,650,77,'purple',opacity=.18)
    d.text(910,101,'DSpark · Backbone 选出 Anchor 后，额外猜五个词',12,True,anchor='start')
    topic('input',908,107,185,31,['[Anchor, Noise, Noise, Noise, Noise]','接收 Features 上下文'],10)
    topic('blocks',1105,107,156,31,['3 Draft Blocks · 每块五位置','mtp[0] → mtp[1] → mtp[2]'],10)
    topic('head',1273,107,114,31,['LM Head','5 组 Base Logits'],10)
    topic('markov',1399,107,143,31,['Markov Head · 逐个选词','候选 A → B → C → D → E'],10)
    for key,l,r in [('slots-blocks',1093,1105),('blocks-head',1261,1273),('head-markov',1387,1399)]:
        link(key,f'M{l} 122 H{r}')
    topic('confidence',1273,145,269,17,['Confidence Head → 分数 → Scheduler'],10)
    link('hidden-confidence','M1183 138 V153 H1273')
    link('markov-confidence','M1470 138 V145')
    d.text(1220,150,'隐藏状态',9)
    link('anchor','M1654 285 H1667 V80 H893 V122 H908')
    d.text(1515,78,'Input A · Anchor Token',10)
    topic('anchor',1572,276,82,18,['Sample → Anchor'],10)
    link('features','M1304 185 V179 H930 V138')
    d.raw('<g class="dspark-topic" data-dspark-topic="features" role="button" tabindex="0" aria-label="Backbone Features">')
    d.raw('<rect class="hit-label" x="935" y="165" width="485" height="13" fill="transparent"/>')
    d.text(1178,175,'Input B · Layers 37/38/39 输入 Hidden States → 拼接与投影 → Context',10)
    d.raw('</g>')
    link('candidates-verification','M1542 122 H1561 V307 H1510 V314',True)
    link('scores-scheduler','M1542 153 H1559 V302 H1450 V314',True)
    topic('verification',1380,314,280,42,['Backbone ×40 Verification → 接受或修正','Anchor + 候选输入 · 通过的留下，继续生成'],11)
    link('verification-next-anchor','M1660 336 H1670 V302 H1613 V294',True)
    d.raw('<g class="dspark-topic" data-dspark-topic="implementation" role="button" tabindex="0" aria-label="Conceptual serving pipeline">')
    d.raw('<rect class="hit-label" x="1380" y="357" width="280" height="8" fill="transparent"/>')
    d.text(1520,363,'虚线：概念 Serving Pipeline；generate.py 未实现',9)
    d.raw('</g></g>')
