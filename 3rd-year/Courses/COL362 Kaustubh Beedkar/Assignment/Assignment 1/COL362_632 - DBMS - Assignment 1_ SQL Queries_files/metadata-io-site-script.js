(function () {
  var accountId;
  var async = false;
  var pollingTimeout = 6000;
  var baseUrl = "https://platformapi.metadata.io";
  var primaryKey = "name";
  var visitorIdKey = "Metadata_visitor_id";
  var cidKey = "metadata_cid";
  var listenFormSubmit = true;
  var onFormInit = function () {}
  var formsSet = new Set();
  var blacklistSet = new Set(["hs_context"]);
  var log = { sentData: [], errors: [], formsSet }
  var adjustDataBeforeSend = function (data) {
    return data;
  }

  function getCookieValue(key) {
    var cookie = document.cookie.split("; ").find(function (cookie) {
      return cookie.indexOf(key) === 0;
    });

    if (cookie) {
      return cookie.split("=")[1];
    }
  }

  function getUrlParameter(parameter, fallback) {
    var result = new RegExp("[\\?&]" + parameter + "=([^&#]*)").exec(
      window.location.search
    );

    return result === null
      ? fallback
      : decodeURIComponent(result[1].replace(/\+/g, " "));
  }

  function normalizeFormData(data) {
    var acc = {}

    if (!data) {
      return acc;
    }

    Object.entries(data).forEach(function (entry) {
      var key = entry[0];
      var value = entry[1];

      if (!blacklistSet.has(key)) {
        acc[key] = value;
      }
    });

    return acc;
  }

  function getFormData(formEl) {
    var data = {}
    var elements = formEl.querySelectorAll(
      "input[name]:not([type=password]):not([type=hidden]),select[name]"
    );

    Array.prototype.forEach.call(elements, function (e) {
      var key = e[primaryKey] || e.name;
      data[key] = e.value;
    });

    return data;
  }

  function getAllFields() {
    var acc = [];

    formsSet.forEach(function (formEl) {
      acc = acc.concat(Object.keys(getFormData(formEl)));
    });

    return acc;
  }

  function sendData(sfv, callback) {
    var data = adjustDataBeforeSend({
      cid: getUrlParameter("cid"),
      lpu: window.location.href,
      account_id: accountId,
      visitor_id: getCookieValue(visitorIdKey),
      metadata_cid: getUrlParameter(cidKey, getCookieValue(cidKey)),
      url_referrer: window.document.referrer,
      sfv: normalizeFormData(sfv)
    });

    var xhr = new XMLHttpRequest();

    // TODO: consider using fetch api with "keepalive: true"
    // when it's widely supported instead of sync request
    xhr.open("POST", baseUrl + "/insight", async);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onload = function () {
      if (xhr.status >= 200 && xhr.status <= 299) {
        callback && callback();
      }
    }

    xhr.send(JSON.stringify(data));
    log.sentData.push(data);
  }

  function sendFormData(formEl, callback) {
    sendData(getFormData(formEl), callback);
  }

  function listenHubspotCallback() {
    var interval;
    var handler = function () {
      if (window.hubspot) {
        var onMessage = function (e) {
          if (
            e.data.type === "hsFormCallback" &&
            e.data.eventName === "onFormSubmit"
          ) {
            var formEl = window.document.querySelector(
              'form[data-form-id="' + e.data.id + '"]'
            );

            if (formEl) {
              sendFormData(formEl);
            } else {
              // an iframe form, fallback to hs provided data
              sendData(
                e.data.data.reduce(function (acc, field) {
                  acc[field.name] = field.value;
                  return acc;
                }, {})
              );
            }
          }
        }

        window.addEventListener("message", onMessage, false);
        window.clearTimeout(interval);
      }
    }

    interval = window.setInterval(handler, pollingTimeout);
    handler();
  }

  function listenHubspotCalendarCallback() {
    var callback = function (e) {
      if (e.data && e.data.meetingBookSucceeded) {
        sendData(e.data.meetingsPayload.bookingResponse.postResponse.contact);
      }
    }

    window.addEventListener("message", callback);
  }

  function listenMarketoCallback() {
    var interval;
    var handler = function () {
      if (window.MktoForms2 && window.MktoForms2.whenReady) {
        var onReady = function (form) {
          form.onSuccess(function () {
            sendFormData(form.getFormElem().get(0));
          });
        }

        window.MktoForms2.whenReady(onReady);
        window.clearTimeout(interval);
      }
    }

    interval = window.setInterval(handler, pollingTimeout);
    handler();
  }

  function listenDriftCallback() {
    var interval;
    var handler = function () {
      if (window.drift && window.drift.on) {
        var onEmailCapture = function (e) {
          sendData({ email: e.data.email });
        }

        window.drift.on("emailCapture", onEmailCapture);
        window.clearTimeout(interval);
      }
    }

    interval = window.setInterval(handler, pollingTimeout);
    handler();
  }

  function listenQualifiedCallback() {
    var interval;
    var handler = function () {
      if (window.qualified) {
        var callback = function (name, data) {
          if (name === "Meeting Booked") {
            sendData(data.field_values);
          }
        }

        window.qualified("handleEvents", callback);
        window.clearTimeout(interval);
      }
    }

    interval = window.setInterval(handler, pollingTimeout);
    handler();
  }

  function listenPardotIframeFormCallback() {
    var callback = function (e) {
      if (e.data) {
        var data;

        try {
          data = JSON.parse(e.data);
        } catch (e) {
          data = {}
        }

        if (data.event === "ContactUsCallback") {
          sendData(data, function () {
            e.source.postMessage("SendDataCallback", e.origin);

            // notion patched
            const url = new URL(data.redirectUrl);
            if (["https:", "http:"].includes(url.protocol.toLowerCase())) {
              location.href = data.redirectUrl;
            }
          });
        }
      }
    }

    window.addEventListener("message", callback);
  }

  function submitHandler(e) {
    var form = e.target;
    var id = form.getAttribute("id") || "";

    if (
      id.search("hsForm_") === 0 ||
      (id.search("mktoForm_") === 0 && window.MktoForms2)
    ) {
      return;
    }

    sendData(getFormData(form));
  }

  function initForm(el) {
    if (formsSet.has(el)) {
      return;
    }

    if (listenFormSubmit) {
      el.addEventListener("submit", submitHandler);
    }

    formsSet.add(el);
    onFormInit(el);
  }

  function observer(mutations) {
    mutations.forEach(function (mutation) {
      mutation.addedNodes.forEach(function (node) {
        if (node.tagName === "FORM") {
          initForm(node);
        } else if (node.querySelectorAll) {
          Array.prototype.forEach.call(node.querySelectorAll("form"), initForm);
        }
      });
    });
  }

  function init(options) {
    if (options.baseUrl) {
      baseUrl = options.baseUrl;
    }

    if (options.accountId) {
      accountId = options.accountId;
    }

    if (options.adjustDataBeforeSend) {
      adjustDataBeforeSend = options.adjustDataBeforeSend;
    }

    if (options.primaryKey) {
      primaryKey = options.primaryKey;
    }

    if (typeof options.async === "boolean") {
      async = options.async;
    }

    if (typeof options.listenFormSubmit === "boolean") {
      listenFormSubmit = options.listenFormSubmit;
    }

    if (options.onFormInit) {
      onFormInit = options.onFormInit;
    }

    if (options.initForms !== false) {
      listenHubspotCallback();
      listenHubspotCalendarCallback();
      listenMarketoCallback();
      listenDriftCallback();
      listenQualifiedCallback();
      listenPardotIframeFormCallback();

      Array.prototype.forEach.call(
        window.document.querySelectorAll("form"),
        initForm
      );

      new MutationObserver(observer).observe(window.document.body, {
        childList: true,
        subtree: true
      });
    }

    if (options.onReady) {
      options.onReady();
    }
  }

  window.Metadata = window.Metadata || {}
  window.Metadata.siteScript = {
    init,
    sendData,
    sendFormData,
    getAllFields,
    log
  }
})();
