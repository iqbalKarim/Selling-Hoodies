const base = "http://127.0.0.1:5000/"

export function make_param_string(params) {
  if (typeof params != "object") {
    throw new Error("Params must be in the form of an Object!")
  } else {
    var param_string = "?"
    for (const [key, value] of Object.entries(params)) {
      param_string += `${key}=${value}&`
    }
    param_string = param_string.slice(0, -1)
  }
  return param_string
}

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

async function post_request(url = "", body) {
  let headers = new Headers()
  headers.append("Content-Type", "application/json")
  headers.append("Accept", "application/json")
  headers.append("Access-Control-Allow-Origin", "*")
  headers.append("Access-Control-Allow-Headers", "*")
  headers.append("Access-Control-Allow-Methods", "*")

  try {
    const res = await fetch(base + url, { method: "POST", body: JSON.stringify(body), headers: headers })
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

export function get_emnist_images(params) {
  let param_str = make_param_string(params)
  return get_request("emnist" + param_str)
}

export function get_mnist_images(params) {
  let param_str = make_param_string(params)
  return get_request("mnist" + param_str)
}

export function get_graffiti_images(params) {
  let param_str = make_param_string(params)
  return get_request("graffiti" + param_str)
}

export function get_met_images(params) {
  let param_str = make_param_string(params)
  return get_request("metfaces" + param_str)
}

export function use_nst(body) {
  return post_request("nst", body)
}
