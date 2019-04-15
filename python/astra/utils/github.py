
import os
from astra import config
import requests

def graphql(query_string, token=None):
    r"""
    Execute a GraphQL query on GitHub.

    :param query_string:
        The query string to execute.
    
    :param token: [optional]
        The GitHub personal access token to use for this query. If `None` is
        given then this will default to the access token called "github.token"
        in ``astra.config``.
    """

    if token is None:
        tokens = [
            config.get("github.token", None),
            os.getenv("ASTRA_GITHUB_TOKEN", None)
        ]
        for token in tokens:
            if token is not None: break

        else:
            raise ValueError("no github.token key found in Astra configuration"\
                             " or ASTRA_GITHUB_TOKEN environment variable")

    headers = dict([("Authorization", f"token {token}")])
    r = requests.post("https://api.github.com/graphql",
                      json=dict(query=query_string), headers=headers)

    if not r.ok:
        r.raise_for_status()

    return r.json()


def get_repository_summary(owner, repository, **kwargs):
    r"""
    Return summary information about a GitHub repository using the GitHub
    GraphQL API.

    :param owner:
        The GitHub username of the owner.

    :param repository:
        The name of the repository.
    """

    q = """
    query {{
      repository(owner: "{owner}", name: "{repository}") {{
        description
        descriptionHTML
        shortDescriptionHTML
        isPrivate
        pushedAt
        sshUrl
        createdAt
        updatedAt
        url
        owner {{
          id
          __typename
          login
        }}
      }}
    }}
    """.format(owner=owner, repository=repository)

    content = graphql(q, **kwargs)
    return content["data"]["repository"]


def get_most_recent_release(owner, repository, **kwargs):
    r"""
    Query the GitHub GraphQL API to access information about the most recent
    release for the given ``owner`` and ``repository``.

    :param owner:
        The GitHub username of the owner.

    :param repository:
        The name of the repository.
    """

    q = """
    query {{
      repository(owner: "{owner}", name: "{repository}") {{
        releases(last: 1) {{
          edges {{
            node {{
              url
              tag {{
                ...refInfo
              }}
            }}
          }}
        }}
      }}
    }}

    fragment refInfo on Ref {{
      name
      target {{
        sha: oid
        __typename
        ... on Tag {{
          target {{
            ... on Commit {{
              ...commitInfo
              commitResourcePath
            }}
          }}
          tagger {{
            name
            email
            date
          }}
        }}
        ... on Commit {{
          ...commitInfo
        }}
      }}
    }}

    fragment commitInfo on Commit {{
      message
      zipballUrl
      tarballUrl
      committedDate
      author {{
        name
        email
        date
      }}
    }}
    """.format(owner=owner, repository=repository)
    
    content = graphql(q, **kwargs)
    edges = content["data"]["repository"]["releases"]["edges"]
    if len(edges):
        return edges[0]["node"]["tag"]
    else:
        # No releases
        return dict()
