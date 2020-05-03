import requests
import os


def get_overview():
    # Use Github Rest API to get releases:
    release_dict = requests.get('https://api.github.com/repos/do-mpc/do-mpc/releases').json()




    text = ''
    text += '# Release notes'
    text += '\n'
    text += 'This content is autogenereated from our Github [release notes](https://github.com/do-mpc/do-mpc/releases).'
    text += '\n'

    for release_i in release_dict:
        name_i = release_i['name']
        body_i = release_i['body']
        body_i = body_i.replace('# ', '### ')
        print(name_i)

        text += '## {}'.format(name_i)
        text += '\n'
        text += body_i
        text += '\n'

    with open('release_notes.md', 'w') as f:
        f.write(text)
