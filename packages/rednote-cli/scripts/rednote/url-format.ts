export function decodeUrlEscapedValue(value: string | null | undefined) {
  if (typeof value !== 'string' || value.length === 0) {
    return null;
  }

  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

function getExploreNoteId(url: URL) {
  const segments = url.pathname.split('/').filter(Boolean);
  return segments.at(-1) ?? null;
}

export function buildExploreUrl(noteId: string, xsecToken?: string | null) {
  const normalizedToken = decodeUrlEscapedValue(xsecToken);
  return normalizedToken
    ? `https://www.xiaohongshu.com/explore/${noteId}?xsec_token=${normalizedToken}`
    : `https://www.xiaohongshu.com/explore/${noteId}`;
}

export function normalizeExploreUrlForOutput(url: string) {
  try {
    const parsed = new URL(url);
    const noteId = getExploreNoteId(parsed);
    if (!noteId) {
      return url;
    }

    return buildExploreUrl(noteId, parsed.searchParams.get('xsec_token'));
  } catch {
    return url;
  }
}

export function normalizeExploreUrlForFetch(url: string) {
  try {
    const parsed = new URL(url);
    if (!parsed.searchParams.has('xsec_source')) {
      parsed.searchParams.set('xsec_source', 'pc_feed');
    }
    return parsed.toString();
  } catch {
    return url;
  }
}
