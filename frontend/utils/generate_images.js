const base = "http://127.0.0.1:5000/"

async function get_request(url = "") {
  let headers = new Headers()
  headers.append("Content-Type", "application/json")
  headers.append("Accept", "application/json")
  headers.append("Access-Control-Allow-Origin", "*")
  headers.append("Access-Control-Allow-Headers", "*")
  headers.append("Access-Control-Allow-Methods", "*")

  try {
    const res = await fetch(base + url, { headers: headers })
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
  return get_request("?normalize=False")
}

export function get_emnist_images() {
  return get_request("emnist")
}

export function get_mnist_images() {
  return get_request("mnist")
}

export function get_graffiti_images() {
  return get_request("graffiti")
}
