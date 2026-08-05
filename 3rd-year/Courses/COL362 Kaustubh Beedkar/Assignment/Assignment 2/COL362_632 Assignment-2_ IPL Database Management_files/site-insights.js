(function () {
  /**
   * @type {string} Key for the visitor ID cookie.
   */
  const visitorIdKey = "Metadata_visitor_id";

  /**
   * @type {string} Key for the session ID cookie.
   */
  const sessionIdKey = "Metadata_session_id";

  /**
   * @type {string} IP address of the client.
   */
  let ip;

  /**
   * Account configuration object.
   */
  const config = {
    invalid: true
  };

  /**
   * Options object.
   */
  const opts = {
    /**
     * @type {string} Base URL for the CDN.
     */
    cdnBaseUrl: "https://cdn.metadata.io/pixel/config",

    /**
     * @type {string} Base URL for the API.
     */
    baseUrl: "https://api-gw.metadata.io",

    /**
     * @type {string} Account ID.
     */
    accountId: null
  };

  /**
   * Get the value of a cookie.
   * @param {string} key - The key of the cookie.
   * @returns {string|null} The value of the cookie, or null if not found.
   */
  const getCookieValue = (key) => {
    const cookie = document.cookie.split("; ").find(function (cookie) {
      return cookie.indexOf(key) === 0;
    });

    if (cookie) {
      return cookie.split("=")[1];
    }

    return null;
  };

  /**
   * Set the value of a cookie.
   * @param {string} key - The key of the cookie.
   * @param {string} value - The value to set.
   * @param {string} expires - The expiration date as a GMT string.
   */
  const setCookieValue = (key, value, expires) => {
    document.cookie = key + "=" + value + "; expires=" + expires + "; path=/";
  };

  /**
   * Create a unique ID.
   * @returns {string} The created ID.
   */
  const createId = () => {
    return Date.now().toString(36) + Math.random().toString(36).substring(2);
  };

  /**
   * Create a cookie expiration date.
   * @param {number} minutes - The number of minutes until the cookie expires.
   * @returns {string} The expiration date as a GMT string.
   */
  const createCookieExpiration = (minutes) => {
    return new Date(new Date().getTime() + 1000 * 60 * minutes).toGMTString();
  };

  /**
   * @type {string} Visitor ID, retrieved from a cookie or generated if not present.
   */
  const visitorId = (function () {
    const storedVisitorId = getCookieValue(visitorIdKey);

    if (storedVisitorId) {
      return storedVisitorId;
    }

    const visitorId = createId();
    const expires = createCookieExpiration(525600); // 1 year from now

    setCookieValue(visitorIdKey, visitorId, expires);
    return visitorId;
  })();

  /**
   * @type {string} Session ID, retrieved from a cookie or generated if not present.
   */
  const sessionId = (function () {
    const storedSessionId = getCookieValue(sessionIdKey);

    if (storedSessionId) {
      return storedSessionId;
    }

    const sessionId = createId();
    const expires = createCookieExpiration(30);

    setCookieValue(sessionIdKey, sessionId, expires);
    return sessionId;
  })();

  /**
   * Decode a JWT token.
   * @param {string} token - The JWT token to decode.
   * @returns {Object} The decoded JWT token.
   */
  const decodeJwt = (token) => {
    const [header, data, signature] = token.split(".");
    const decodedHeader = JSON.parse(window.atob(header));
    const decodedData = JSON.parse(window.atob(data));
    return { header: decodedHeader, data: decodedData, signature };
  };

  /**
   * Resolve the IP address of the client.
   * @returns {Promise<void>} A promise that resolves when the IP has been resolved.
   */
  const resolveIp = () =>
    fetch("https://api.ipify.org?format=json")
      .then((response) => response.json())
      .then((payload) => {
        ip = payload.ip;
      })
      .catch((e) => {
        console.error("Error resolving ip", e);
      });

  /**
   * Resolve the configuration for the site.
   * @returns {Promise<void>} A promise that resolves when the configuration has been resolved.
   */
  const resolveConfig = () =>
    fetch(`${opts.cdnBaseUrl}/${opts.accountId}.json`)
      .then((response) => {
        if (response.status === 200) {
          return response.json();
        }

        return fetch(`${opts.cdnBaseUrl}/default.json`).then((response) => {
          if (response.status === 200) {
            return response.json();
          }

          throw new Error("Both default and account config missing");
        });
      })
      .then((payload) => {
        const now = Date.now() / 1000;
        const decodedJwt = decodeJwt(payload.pixelJwt);

        Object.assign(config, decodedJwt, {
          invalid: decodedJwt.data.exp < now || decodedJwt.data.nbf > now
        });
      })
      .catch((e) => {
        console.error("Error resolving config", e);
      });

  /**
   * Record traffic data.
   * @returns {Promise<void>} A promise that resolves when the traffic data has been recorded.
   */
  const recordTraffic = () => {
    if (config.invalid || !config.data.collectInsights) {
      return;
    }

    fetch(`${opts.baseUrl}/traffic`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ip: ip,
        url: window.location.href,
        url_referrer: window.document.referrer,
        account_id: opts.accountId,
        session_id: sessionId,
        visitor_id: visitorId
      })
    }).catch((e) => {
      console.error("Error recording traffic", e);
    });
  };

  /**
   * Prolong the session.
   * @returns {void}
   */
  const prolongSession = () => {
    const storedSessionId = getCookieValue(sessionIdKey);

    if (storedSessionId) {
      const expires = createCookieExpiration(30);
      setCookieValue(sessionIdKey, storedSessionId, expires);
    }
  };

  /**
   * Throttle a function.
   * @param {Function} func - The function to throttle.
   * @param {number} timeFrame - The minimum time between function calls, in milliseconds.
   * @returns {Function} The throttled function.
   */
  const throttle = (func, timeFrame) => {
    let lastTime = 0;

    return (...args) => {
      const now = new Date();

      if (now - lastTime >= timeFrame) {
        func(...args);
        lastTime = now;
      }
    };
  };

  /**
   * Initialize the site insights.
   * @param {Object} options - The initialization options.
   * @param {string} options.baseUrl - The base URL for the API.
   * @param {string} options.accountId - The account ID.
   * @param {string} options.cdnBaseUrl - The base URL for the CDN.
   */
  const init = (options) => {
    Object.assign(opts, options);

    window.addEventListener("scroll", throttle(prolongSession, 1000));
    window.addEventListener("click", throttle(prolongSession, 1000));

    Promise.all([resolveIp(), resolveConfig()]).finally(() => {
      recordTraffic();
    });
  };

  window.Metadata = window.Metadata || {};
  window.Metadata.siteInsights = { init };
})();
