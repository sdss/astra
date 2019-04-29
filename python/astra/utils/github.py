
import os
from astra import config
from astra.utils import log
import requests


def validate_slug(slug_string):
    r"""
    Validate a given GitHub repository slug, given some input.

    :param slug_string:
        The given slug string, which should be in the form '{OWNER}/{REPO}'.
        If no '{OWNER}' is given, it will be assumed to be owned by the SDSS organization.
    """

    slug_string = f"{slug_string}".strip().lower()
    if "/" not in slug_string:
        log.info(f"Assuming GitHub repository '{slug_string}' is owned by SDSS (sdss/{slug_string})")
        slug_string = f"sdss/{slug_string}"
    return slug_string

def validate_repository_name(repository_name):
    if "/" in repository_name:
      raise ValueError("repository name cannot contain forward slashes ('/')")
    return repository_name.strip().lower()
    

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
        # TODO: Consider using SDSS_GITHUB_KEY because that is used by `sdss_install`
        #       See https://github.com/sdss/sdss_install/blob/master/python/sdss_install/application/Client.py
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


def get_most_recent_release(owner, repository, n=1, **kwargs):
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
        releases(last: {n}) {{
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
    """.format(owner=owner, n=n, repository=repository)
    
    content = graphql(q, **kwargs)
    edges = content["data"]["repository"]["releases"]["edges"]
    if len(edges):
        if n == 1:
            return edges[0]["node"]["tag"]
        return [edges[i]["node"]["tag"] for i in range(min(n, len(edges)))]
          
    else:
        # No releases
        return dict()
