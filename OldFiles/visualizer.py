import webbrowser

def score_to_color(score):
    color = ""
    if score > .5:
        hue = (1- (score-.5)*2) * 100
        color = 'hsl(' + str(hue) + ", 50%, 50%)"
    return color

def score_to_color_alpha(score):
    color = ""
    if score > .5:
        lightness = (abs(score-1)+.5)*100
        color = 'hsl(120, 100%,' + str(lightness) + '%)'
    else:
        lightness = (score+0.5)*100
        color = 'hsl(0, 100%,' + str(lightness) + '%)'

    return color

def to_html(lst):
    
    body = '<p>'

    head = '<!doctype html><head><meta charset="utf-16"><title>QuoteBERT in Action!</title> \
        <meta name="description" content="Prototype of Quote Selection using BERT"> \
        <meta name="author" content="Lasse Funder Andersen">\
        </head><body><p style="font-family:arial;font-size:16px>'

    closing = '</p></body>'

    for q, s in lst:
        body = body + '<span style=\"background-color:' + score_to_color_alpha(s) +';\">' + q + ' </span>'

    html = head + body + closing
    with open('test.html', 'w', encoding='utf-8') as file:      
        file.write(html)
    
    webbrowser.open('test.html')


strings = ['sætning nummer 1. ', 'sætning nummer 2', 'sætning nummer 3', 'sætning nummer 4']
        
# for testing
if __name__ == "__main__":
    empty = "_________"
    quotes = [['I år var det 50 år siden, at mennesket landede på månen og vi fik vores egen planet Jorden at se som en lille klode i det store rum: ganske alene, men så smuk og rund og blå: Planeten, hvor vi har hjemme.', 0.99533694679188373], ['For os her i Danmark er det måske ikke så overraskende, at planeten er blå, for vi har jo havet foran os og den blå himmel over os.', 0.06841709123095198], ['Så storslået og varieret vor Jord end kan synes, er den dog sårbar.', 0.45], ['Det er vi ved at lære at indse, og det kan godt bekymre, ikke mindst mange unge, som ser klimaforandringerne, der gør sig tydeligt gældende i disse år.', 0.55]]
    legend = [['JUICY', 1],[empty, .9],[empty, .8],[empty, .7],[empty, .6],[empty, .5],[empty, .4],[empty, .3],[empty, .2], [empty, .1], ['BORING', 0]]    

    to_html(legend)

