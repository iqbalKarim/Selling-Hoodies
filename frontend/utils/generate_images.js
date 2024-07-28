const url = "http://127.0.0.1:5000/"

async function get_request() {
  let headers = new Headers()
  headers.append("Content-Type", "application/json")
  headers.append("Accept", "application/json")
  headers.append("Access-Control-Allow-Origin", "*")

  try {
    const res = await fetch(url, { headers: headers })
    if (!res.ok) {
      throw new Error(`Response status from inside: ${res.status}`)
    }
    const json = await res.json()
    return json
  } catch (error) {
    return { error: error.message }
  }
}

export function get_image() {
  return get_request()
}
