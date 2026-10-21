"""Compact, connected DSpark branch in the existing top architecture panel."""
import build_diagram as d


def topic(key,x,y,w,h,lines,color='purple',size=12):
    d.raw(f'<g class="dspark-topic" data-dspark-topic="{key}" role="button" tabindex="0" aria-label="DSpark {key}">')
    d.rect(x,y,w,h,color)
    d.text(x+w/2,y+h/2-(len(lines)-1)*size*.55+size*.33,lines,size,leading=size*1.1,maxwidth=w-8)
    d.raw('</g>')


def draw():
    d.raw('<g id="dspark-overview" data-dspark-reference="model.forward + forward_spec" font-family="Times New Roman, Microsoft YaHei, serif">')
    # Keep every pre-existing backbone box and connection in its original location.
    d.rect(900,87,650,76,'purple',opacity=.18)
    d.text(910,103,'DSpark Draft Branch · 3 extra blocks × 5 positions',15,True,anchor='start')
    d.raw('<g class="dspark-topic" data-dspark-topic="implementation" role="button" tabindex="0" aria-label="DSpark implementation status">')
    d.raw('<rect class="hit-label" x="1285" y="89" width="257" height="16" fill="transparent"/>')
    d.text(1540,102,'forward_spec: implemented · serving loop: conceptual',10,anchor='end',maxwidth=255)
    d.raw('</g>')
    topic('input',908,113,140,42,['Input Builder','Anchor + 4 Noise'])
    topic('blocks',1060,113,185,42,['5 positions through all 3 blocks','mtp[0] → mtp[1] → mtp[2]'],size=11.5)
    topic('markov',1257,113,115,42,['Base logits →','Markov Head (serial)'],size=11)
    topic('confidence',1384,113,158,42,['5 draft candidates','Confidence Head → scores'],size=12)
    for left,right in [(1048,1060),(1245,1257),(1372,1384)]:d.arrow(f'M{left} 134 H{right-1}',width=1.2)
    # A sampled Backbone token feeds the builder, rather than emerging from DSpark.
    d.raw('<g data-dspark-link="anchor">')
    d.arrow('M1660 216 H1666 V80 H894 V134 H908',width=1.3)
    d.text(1595,75,'Anchor Token',11,True)
    d.raw('</g>')
    d.raw('<g class="dspark-topic" data-dspark-topic="anchor" role="button" tabindex="0" aria-label="Anchor token from Backbone">')
    d.raw('<rect class="hit-label" x="1570" y="275" width="86" height="20" fill="transparent"/>')
    d.text(1613,288,'Select → Anchor',11,True,maxwidth=84)
    d.raw('</g>')
    # Grouped tap denotes features BEFORE the selected deep Backbone layers.
    d.raw('<g class="dspark-topic" data-dspark-topic="features" role="button" tabindex="0" aria-label="Backbone features before layers 37 38 39">')
    d.arrow('M1304 185 V178 H934 V155','#8d4bc6',width=1.2)
    d.raw('<rect class="hit-label" x="963" y="163" width="335" height="14" fill="transparent"/>')
    d.text(1130,173,'Before 37 / 38 / 39 → concat → main_proj → context',10.5,maxwidth=331)
    d.raw('</g>')
    # Dashed links refer to a serving-level consumer, not another mandatory layer.
    d.raw('<g data-dspark-link="verification">')
    d.arrow('M1463 155 V164 H1483 V305 H1520 V314','#8d4bc6',True,1.1)
    d.raw('</g>')
    topic('verification',1380,314,280,42,['Confidence → Scheduler → Target ×40','Verification / Accept-Reject · conceptual'],size=11.5)
    d.raw('</g>')
