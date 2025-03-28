export function getPath(path) {
  // Ensure path starts with a slash and remove any duplicate slashes
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${import.meta.env.BASE_URL}${cleanPath}`;
}