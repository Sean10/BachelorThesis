from flask import render_template, request, session, redirect, flash, url_for, abort


from app import create_app
from app.nlp import nlp
from config import config


app = create_app(config)
model = nlp()


@app.route('/', methods=['GET'])
def index():
    '''

    :return:
    '''

    return render_template('index.html', meta_title='情感分析'),200


@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html', meta_title='404'), 404

@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    if request.method == 'POST':
        text = request.form.get('textarea')

        ans = model.analysis(text)
        flash(ans)
        return render_template('index.html', meta_title='情感分析')



app.run(host="0.0.0.0", port=5000, debug=app.config['DEBUG'])